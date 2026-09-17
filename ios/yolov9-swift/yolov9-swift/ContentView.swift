import SwiftUI
import AVFoundation
import CoreImage

let yoloClasses: [(name: String, color: UIColor)] = [
    ("person", .red),
    ("bicycle", .green),
    ("car", .blue),
    ("motorcycle", .cyan),
    ("airplane", .magenta),
    ("bus", .yellow),
    ("train", .orange),
    ("truck", .purple),
    ("boat", .brown),
    ("traffic light", UIColor(red: 0.5, green: 0.7, blue: 0.2, alpha: 1.0)),
    ("fire hydrant", UIColor(red: 0.8, green: 0.1, blue: 0.1, alpha: 1.0)),
    ("stop sign", UIColor(red: 0.3, green: 0.3, blue: 0.8, alpha: 1.0)),
    ("parking meter", UIColor(red: 0.7, green: 0.5, blue: 0.3, alpha: 1.0)),
    ("bench", UIColor(red: 0.4, green: 0.4, blue: 0.2, alpha: 1.0)),
    ("bird", UIColor(red: 0.1, green: 0.5, blue: 0.9, alpha: 1.0)),
    ("cat", UIColor(red: 0.8, green: 0.2, blue: 0.6, alpha: 1.0)),
    ("dog", UIColor(red: 0.9, green: 0.3, blue: 0.3, alpha: 1.0)),
    ("horse", UIColor(red: 0.2, green: 0.6, blue: 0.7, alpha: 1.0)),
    ("sheep", UIColor(red: 0.7, green: 0.3, blue: 0.5, alpha: 1.0)),
    ("cow", UIColor(red: 0.4, green: 0.8, blue: 0.4, alpha: 1.0)),
    ("elephant", UIColor(red: 0.3, green: 0.4, blue: 0.9, alpha: 1.0)),
    ("bear", UIColor(red: 0.6, green: 0.2, blue: 0.8, alpha: 1.0)),
    ("zebra", UIColor(red: 0.8, green: 0.5, blue: 0.2, alpha: 1.0)),
    ("giraffe", UIColor(red: 0.5, green: 0.9, blue: 0.1, alpha: 1.0)),
    ("backpack", UIColor(red: 0.3, green: 0.7, blue: 0.4, alpha: 1.0)),
    ("umbrella", UIColor(red: 0.4, green: 0.6, blue: 0.9, alpha: 1.0)),
    ("handbag", UIColor(red: 0.9, green: 0.2, blue: 0.5, alpha: 1.0)),
    ("tie", UIColor(red: 0.5, green: 0.3, blue: 0.7, alpha: 1.0)),
    ("suitcase", UIColor(red: 0.6, green: 0.7, blue: 0.2, alpha: 1.0)),
    ("frisbee", UIColor(red: 0.7, green: 0.2, blue: 0.4, alpha: 1.0)),
    ("skis", UIColor(red: 0.3, green: 0.9, blue: 0.3, alpha: 1.0)),
    ("snowboard", UIColor(red: 0.8, green: 0.1, blue: 0.6, alpha: 1.0)),
    ("sports ball", UIColor(red: 0.4, green: 0.3, blue: 0.8, alpha: 1.0)),
    ("kite", UIColor(red: 0.2, green: 0.5, blue: 0.7, alpha: 1.0)),
    ("baseball bat", UIColor(red: 0.6, green: 0.4, blue: 0.2, alpha: 1.0)),
    ("baseball glove", UIColor(red: 0.7, green: 0.1, blue: 0.4, alpha: 1.0)),
    ("skateboard", UIColor(red: 0.5, green: 0.8, blue: 0.5, alpha: 1.0)),
    ("surfboard", UIColor(red: 0.8, green: 0.3, blue: 0.6, alpha: 1.0)),
    ("tennis racket", UIColor(red: 0.2, green: 0.7, blue: 0.9, alpha: 1.0)),
    ("bottle", UIColor(red: 0.9, green: 0.2, blue: 0.3, alpha: 1.0)),
    ("wine glass", UIColor(red: 0.6, green: 0.6, blue: 0.3, alpha: 1.0)),
    ("cup", UIColor(red: 0.3, green: 0.4, blue: 0.9, alpha: 1.0)),
    ("fork", UIColor(red: 0.4, green: 0.7, blue: 0.2, alpha: 1.0)),
    ("knife", UIColor(red: 0.8, green: 0.2, blue: 0.5, alpha: 1.0)),
    ("spoon", UIColor(red: 0.6, green: 0.3, blue: 0.7, alpha: 1.0)),
    ("bowl", UIColor(red: 0.2, green: 0.8, blue: 0.4, alpha: 1.0)),
    ("banana", UIColor(red: 0.7, green: 0.7, blue: 0.1, alpha: 1.0)),
    ("apple", UIColor(red: 0.9, green: 0.1, blue: 0.4, alpha: 1.0)),
    ("sandwich", UIColor(red: 0.4, green: 0.5, blue: 0.8, alpha: 1.0)),
    ("orange", UIColor(red: 0.8, green: 0.6, blue: 0.2, alpha: 1.0)),
    ("broccoli", UIColor(red: 0.3, green: 0.8, blue: 0.3, alpha: 1.0)),
    ("carrot", UIColor(red: 0.7, green: 0.2, blue: 0.6, alpha: 1.0)),
    ("hot dog", UIColor(red: 0.9, green: 0.3, blue: 0.5, alpha: 1.0)),
    ("pizza", UIColor(red: 0.5, green: 0.3, blue: 0.8, alpha: 1.0)),
    ("donut", UIColor(red: 0.8, green: 0.1, blue: 0.4, alpha: 1.0)),
    ("cake", UIColor(red: 0.7, green: 0.5, blue: 0.1, alpha: 1.0)),
    ("chair", UIColor(red: 0.6, green: 0.2, blue: 0.4, alpha: 1.0)),
    ("couch", UIColor(red: 0.4, green: 0.6, blue: 0.2, alpha: 1.0)),
    ("potted plant", UIColor(red: 0.8, green: 0.4, blue: 0.5, alpha: 1.0)),
    ("bed", UIColor(red: 0.3, green: 0.7, blue: 0.7, alpha: 1.0)),
    ("dining table", UIColor(red: 0.5, green: 0.8, blue: 0.3, alpha: 1.0)),
    ("toilet", UIColor(red: 0.7, green: 0.4, blue: 0.6, alpha: 1.0)),
    ("tv", UIColor(red: 0.9, green: 0.5, blue: 0.2, alpha: 1.0)),
    ("laptop", UIColor(red: 0.6, green: 0.3, blue: 0.7, alpha: 1.0)),
    ("mouse", UIColor(red: 0.2, green: 0.9, blue: 0.5, alpha: 1.0)),
    ("remote", UIColor(red: 0.8, green: 0.4, blue: 0.3, alpha: 1.0)),
    ("keyboard", UIColor(red: 0.3, green: 0.6, blue: 0.8, alpha: 1.0)),
    ("cell phone", UIColor(red: 0.7, green: 0.3, blue: 0.9, alpha: 1.0)),
    ("microwave", UIColor(red: 0.4, green: 0.9, blue: 0.4, alpha: 1.0)),
    ("oven", UIColor(red: 0.5, green: 0.7, blue: 0.2, alpha: 1.0)),
    ("toaster", UIColor(red: 0.9, green: 0.2, blue: 0.3, alpha: 1.0)),
    ("sink", UIColor(red: 0.6, green: 0.8, blue: 0.3, alpha: 1.0)),
    ("refrigerator", UIColor(red: 0.8, green: 0.4, blue: 0.7, alpha: 1.0)),
    ("book", UIColor(red: 0.3, green: 0.5, blue: 0.9, alpha: 1.0)),
    ("clock", UIColor(red: 0.7, green: 0.7, blue: 0.2, alpha: 1.0)),
    ("vase", UIColor(red: 0.9, green: 0.4, blue: 0.5, alpha: 1.0)),
    ("scissors", UIColor(red: 0.2, green: 0.7, blue: 0.8, alpha: 1.0)),
    ("teddy bear", UIColor(red: 0.6, green: 0.3, blue: 0.9, alpha: 1.0)),
    ("hair drier", UIColor(red: 0.8, green: 0.2, blue: 0.3, alpha: 1.0)),
    ("toothbrush", UIColor(red: 0.4, green: 0.7, blue: 0.6, alpha: 1.0))
]

var fpsLabel: UILabel?
var lastFrameTime: CFTimeInterval = 0
var frameCount = 0

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var buffers: [Int: MTLBuffer] = [:]
var buffer_sz: [Int: Int] = [:]
var programs: [String: MTLComputePipelineState] = [:]
var yolo_graph: GraphRunner!

func setupFPSLabel(_ view: UIView) {
    let label = UILabel(frame: CGRect(x: 10, y: 60, width: 120, height: 28))
    label.backgroundColor = UIColor(white: 0, alpha: 0.5)
    label.textColor = .white
    label.font = .boldSystemFont(ofSize: 16)
    label.text = "FPS: 0"
    view.addSubview(label)
    fpsLabel = label
}

func updateFPS() {
    let now = CACurrentMediaTime()
    if lastFrameTime > 0 {
        frameCount += 1
        let dt = now - lastFrameTime
        if dt >= 1.0 {
            fpsLabel?.text = String(format: "FPS: %.1f", Double(frameCount) / dt)
            frameCount = 0
            lastFrameTime = now
        }
    } else {
        lastFrameTime = now
    }
}

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
        DispatchQueue.main.async { updateFPS() }
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
        setupFPSLabel(view)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

func clearBoxes() {
    guard let view = previewUIView else { return }
    for sub in view.layer.sublayers ?? [] where sub.name == "rect" {
        sub.removeFromSuperlayer()
    }
    for sub in view.subviews where sub is UILabel && sub !== fpsLabel { sub.removeFromSuperview() }
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

    let font = UIFont.boldSystemFont(ofSize: 11)

    for d in dets {
        let x1 = videoX + ((CGFloat(d[0]) - g_ox) / g_scale / CGFloat(g_rotW)) * videoW
        let y1 = videoY + ((CGFloat(d[1]) - g_oy) / g_scale / CGFloat(g_rotH)) * videoH
        let x2 = videoX + ((CGFloat(d[2]) - g_ox) / g_scale / CGFloat(g_rotW)) * videoW
        let y2 = videoY + ((CGFloat(d[3]) - g_oy) / g_scale / CGFloat(g_rotH)) * videoH

        let rect = CGRect(x: min(x1, x2), y: min(y1, y2),
                          width: abs(x2 - x1), height: abs(y2 - y1))

        let classIdx = Int(d[5])
        let name: String
        let color: UIColor
        if classIdx >= 0 && classIdx < yoloClasses.count {
            name = yoloClasses[classIdx].name
            color = yoloClasses[classIdx].color
        } else {
            name = "\(classIdx)"
            color = .green
        }

        // Box
        let shape = CAShapeLayer()
        shape.name = "rect"
        shape.path = UIBezierPath(rect: rect).cgPath
        shape.strokeColor = color.cgColor
        shape.fillColor = UIColor.clear.cgColor
        shape.lineWidth = 2
        view.layer.addSublayer(shape)

        // YOLO-style label: "name 87%" tight tag at top-left
        let text = "\(name) \(Int((d[4] * 100).rounded()))%"
        let padX: CGFloat = 4
        let textW = (text as NSString).size(withAttributes: [.font: font]).width
        let labelW = ceil(textW + padX * 2)
        let labelH: CGFloat = 14
        let labelY = max(rect.minY - labelH, 0)

        let label = UILabel(frame: CGRect(x: rect.minX, y: labelY,
                                          width: labelW, height: labelH))
        label.text = text
        label.font = font
        label.textColor = .white
        label.backgroundColor = color
        label.textAlignment = .left
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

    let scale = CGFloat(S) / CGFloat(max(srcH, srcW))
    let newW = Int((CGFloat(srcH) * scale).rounded())
    let newH = Int((CGFloat(srcW) * scale).rounded())
    let ox = (S - newW) / 2
    let oy = (S - newH) / 2

    g_rotW = srcH
    g_rotH = srcW
    g_scale = scale
    g_ox = CGFloat(ox)
    g_oy = CGFloat(oy)

    let dstRowBytes = S * 3
    for y in 0..<newH {
        let rotY = Int(CGFloat(y) / scale)
        for x in 0..<newW {
            let rotX = Int(CGFloat(x) / scale)
            let p = src + (srcH - 1 - rotX) * srcRowBytes + rotY * 4
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

