import SwiftUI
import AVFoundation
import CoreImage

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

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Copy frame into the last copyin buffer of the graph
        copyFrameToYoloBuffer(pixelBuffer)
        
        yolo_graph.run() // test
        
        let out = yolo_graph.copyouts[0]
        let rows = buffer_sz[out]! / 6
        let dets = buffers[out]!.contents().bindMemory(to: Float.self, capacity: buffer_sz[out]!)
        let shaped = (0..<rows).map { Array(UnsafeBufferPointer(start: dets + $0*6, count: 6)) }.filter { $0[4] >= 0.25 }
        print("outputs =",shaped.count, shaped)
        DispatchQueue.main.async { drawBoxes(shaped) }
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

var previewUIView: UIView?

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspect
        previewUIView = view
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

func clearBoxes() {
    guard let view = previewUIView else { return }
    for sub in view.layer.sublayers ?? [] where sub.name == "rect" {
        sub.removeFromSuperlayer()
    }
    for sub in view.subviews where sub is UILabel {
        sub.removeFromSuperview()
    }
}

func drawBoxes(_ dets: [[Float]]) {
    guard let view = previewUIView else { return }
    clearBoxes()

    let W = view.bounds.width
    let H = view.bounds.height

    let videoAspect = CGFloat(g_rotW) / CGFloat(g_rotH)
    let viewAspect = W / H

    let videoW: CGFloat
    let videoH: CGFloat
    let videoX: CGFloat
    let videoY: CGFloat

    if viewAspect > videoAspect {
        videoH = H
        videoW = H * videoAspect
        videoX = (W - videoW) / 2
        videoY = 0
    } else {
        videoW = W
        videoH = W / videoAspect
        videoX = 0
        videoY = (H - videoH) / 2
    }

    for d in dets {
        let x1 = videoX + ((CGFloat(d[0]) - g_ox) / g_scale / CGFloat(g_rotW)) * videoW
        let y1 = videoY + ((CGFloat(d[1]) - g_oy) / g_scale / CGFloat(g_rotH)) * videoH
        let x2 = videoX + ((CGFloat(d[2]) - g_ox) / g_scale / CGFloat(g_rotW)) * videoW
        let y2 = videoY + ((CGFloat(d[3]) - g_oy) / g_scale / CGFloat(g_rotH)) * videoH

        let rect = CGRect(x: min(x1, x2), y: min(y1, y2),
                          width: abs(x2 - x1), height: abs(y2 - y1))

        let shape = CAShapeLayer()
        shape.name = "rect"
        shape.path = UIBezierPath(rect: rect).cgPath
        shape.strokeColor = UIColor.green.cgColor
        shape.fillColor = UIColor.clear.cgColor
        shape.lineWidth = 2
        view.layer.addSublayer(shape)

        let label = UILabel(frame: CGRect(x: rect.minX, y: max(rect.minY - 16, 0), width: 90, height: 16))
        label.text = "\(Int(d[5])): \(Int((d[4] * 100).rounded()))%"
        label.font = .boldSystemFont(ofSize: 12)
        label.textColor = .white
        label.backgroundColor = .green
        view.addSubview(label)
    }
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
            let metaLen: UInt32 = fileData.withUnsafeBytes { $0.load(as: UInt32.self) }
            let metaEnd = 4 + Int(metaLen)
            let metaData = fileData.subdata(in: 4..<metaEnd)

            let json = try? JSONSerialization.jsonObject(with: metaData, options: [])
            let items = json as? [Any]

            for item in items! {
                autoreleasepool {
                    guard let dict = item as? [String: Any],
                          let key = dict.keys.first else {
                        return
                    }

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
                let name = item["name"] as! String
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
                    var value = vals[i]
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

let ciContext = CIContext()
var g_rotW = 0
var g_rotH = 0
var g_scale: CGFloat = 1
var g_ox: CGFloat = 0
var g_oy: CGFloat = 0

func copyFrameToYoloBuffer(_ pixelBuffer: CVPixelBuffer) {
    let outIdx = 374
    guard let dstBuffer = buffers[outIdx] else { return }

    let S = 640
    let dstSize = S * S * 3
    let dst = dstBuffer.contents().bindMemory(to: UInt8.self, capacity: dstSize)

    memset(dst, 0, dstSize)

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
    let src = srcBase.assumingMemoryBound(to: UInt8.self)
    let srcRowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let srcW = CVPixelBufferGetWidth(pixelBuffer)
    let srcH = CVPixelBufferGetHeight(pixelBuffer)

    // Back camera gives landscape buffers -> rotate to portrait.
    let needsRotation = srcW > srcH
    let rotW = needsRotation ? srcH : srcW
    let rotH = needsRotation ? srcW : srcH

    let scale = CGFloat(S) / CGFloat(max(rotW, rotH))
    let newW = Int((CGFloat(rotW) * scale).rounded())
    let newH = Int((CGFloat(rotH) * scale).rounded())
    let ox = (S - newW) / 2
    let oy = (S - newH) / 2

    g_rotW = rotW
    g_rotH = rotH
    g_scale = scale
    g_ox = CGFloat(ox)
    g_oy = CGFloat(oy)

    let dstRowBytes = S * 3
    for y in 0..<newH {
        let rotY = Int(CGFloat(y) / scale)
        for x in 0..<newW {
            let rotX = Int(CGFloat(x) / scale)

            let sx: Int
            let sy: Int
            if needsRotation {
                sx = rotY
                sy = srcH - 1 - rotX
            } else {
                sx = rotX
                sy = rotY
            }

            let p = src + sy * srcRowBytes + sx * 4
            let q = dst + (oy + y) * dstRowBytes + (ox + x) * 3
            q[0] = p[0]
            q[1] = p[1]
            q[2] = p[2]
        }
    }
}

#Preview {
    ContentView()
}

