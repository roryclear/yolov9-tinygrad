import SwiftUI
import AVFoundation

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var buffers: [Int: MTLBuffer] = [:]
var buffer_sz: [Int: Int] = [:]
var programs: [String: MTLComputePipelineState] = [:]
var yolo_graph: GraphRunner!

struct ContentView: View {
    @State private var camera = Camera()

    var body: some View {
        CameraPreview(session: camera.session)
            .ignoresSafeArea()
            .onAppear { camera.start() }
            .onDisappear { camera.stop() }
    }
}

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            DispatchQueue.global().async {
                self.setup()
                self.session.startRunning()
            }
        }
    }

    func stop() {
        session.stopRunning()
    }

    private func setup() {
        yolo_graph = GraphRunner(filename: "graph_0.rc")
        print(yolo_graph.copyins)
        yolo_graph.run() // test
        
        guard session.inputs.isEmpty,
              let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }

        session.beginConfiguration()
        session.sessionPreset = .photo
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "video"))
        session.addOutput(output)

        session.commitConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // frames arrive here
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspect
        layer.frame = UIScreen.main.bounds
        view.layer.addSublayer(layer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

class GraphRunner {
    let filename: String
    var calls: [[String: Any]] = []
    var copyouts: [Int] = []
    var copyins: [Int] = []
    var buffs: Set<Int> = []

    init(filename: String) {
        self.filename = filename
        print("GraphRunner initialized with:", filename)

        guard let url = Bundle.main.url(forResource: filename, withExtension: nil) else {
            print("File not found:", filename)
            return
        }

        autoreleasepool {
            guard let fileData = try? Data(contentsOf: url) else {
                print("Failed reading file:", filename)
                return
            }
            // Format written by the Python side:
            //   [u32 little-endian meta length][meta JSON][blobs...]
            guard fileData.count >= 4 else {
                print("File too short:", fileData.count)
                return
            }
            let metaLen: UInt32 = fileData.withUnsafeBytes { $0.load(as: UInt32.self) }
            let metaEnd = 4 + Int(metaLen)
            guard metaEnd <= fileData.count else {
                print("Invalid meta length:", metaLen, "file size:", fileData.count)
                return
            }
            let metaData = fileData.subdata(in: 4..<metaEnd)

            guard let json = try? JSONSerialization.jsonObject(with: metaData, options: []),
                  let items = json as? [Any] else {
                print("Invalid JSON format")
                return
            }

            for item in items {
                autoreleasepool {
                    guard let dict = item as? [String: Any],
                          let key = dict.keys.first else {
                        return
                    }
                    print("rory key =",key)

                    if key == "buff_alloc" {
                        if let info = dict["buff_alloc"] as? [String: Any],
                           let num = info["num"] as? Int,
                           let size = info["size"] as? Int {
                            buffers[num] = device.makeBuffer(length: size, options: .storageModeShared)
                            buffer_sz[num] = size
                        }

                    } else if key == "copyin" {
                        if let info = dict["copyin"] as? [String: Any],
                           let dest = info["dest"] as? Int,
                           let off  = info["off"]  as? Int,
                           let len  = info["len"]  as? Int,
                           let buffer = buffers[dest] {

                            let start = metaEnd + off
                            let end   = start + len
                            guard end <= fileData.count else {
                                print("copyin out of range", dest, off, len)
                                return
                            }
                            let blob = fileData.subdata(in: start..<end)

                            copyins.append(dest)
                            blob.withUnsafeBytes { src in
                                buffer.contents().copyMemory(
                                    from: src.baseAddress!,
                                    byteCount: len
                                )
                            }
                        }
                    } else if key == "program" {
                        if let info = dict["program"] as? [String: Any],
                           let name = info["name"] as? String,
                           let libString = info["lib"] as? String,
                           let libData = Data(base64Encoded: libString) {

                            let dispatchData = libData.withUnsafeBytes { ptr in
                                DispatchData(bytes: ptr)
                            }

                            if let library = try? device.makeLibrary(
                                data: dispatchData as! dispatch_data_t
                            ),
                            let function = library.makeFunction(name: name),
                            let pipeline = try? device.makeComputePipelineState(
                                function: function
                            ) {
                                programs[name] = pipeline
                            }
                        }

                    } else if key == "call" {
                        if let call = dict["call"] as? [String: Any] {
                            calls.append(call)
                            if let bufs = call["buffers"] as? [Int] {
                                for buff in bufs { buffs.insert(buff) }
                            }
                        }
                    } else if key == "copyout" {
                        if let copyout = dict["copyout"] as? Int {
                            copyouts.append(copyout)
                        }
                    }
                }
            }
        }
    }
    
    func run(vals_dict: [Int: Int]? = nil, globals_dict: [Int: Int]? = [:]) {
        autoreleasepool {
            let commandBuffer = queue.makeCommandBuffer()!
            for (index, item) in self.calls.enumerated() {
                let encoder = commandBuffer.makeComputeCommandEncoder()!
                print(index, "of", self.calls.count)
                let name = item["name"] as! String
                print(name)
                let pipeline = programs[name]!
                
                encoder.setComputePipelineState(pipeline)
                
                let bufferIDs = item["buffers"] as! [Int]
                let offsets = item["buffer_offsets"] as! [Int]
                let vals = item["vals"] as! [Int]
                
                for i in 0..<bufferIDs.count {
                    let buffer = buffers[bufferIDs[i]]!
                    encoder.setBuffer(buffer, offset: offsets[i], index: i)
                }
                
                for i in 0..<vals.count{
                    var value = Int32(vals_dict![vals[i]]!)
                    encoder.setBytes(&value, length: 4, index: i+bufferIDs.count)
                }
                
                let global = item["global_size"] as! [Int]
                let local = item["local_size"] as! [Int]
                
                let threadsPerGrid = MTLSize(
                    width: globals_dict?[global[0]] ?? global[0],
                    height: globals_dict?[global[1]] ?? global[1],
                    depth: globals_dict?[global[2]] ?? global[2]
                )
                
                
                let threadsPerThreadgroup = MTLSize(
                    width: local[0],
                    height: local[1],
                    depth: local[2]
                )
                
                encoder.dispatchThreadgroups(
                    threadsPerGrid,
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
                encoder.endEncoding()
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }
}

#Preview {
    ContentView()
}

