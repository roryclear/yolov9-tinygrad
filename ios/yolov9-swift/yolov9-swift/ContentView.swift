import SwiftUI
import AVFoundation

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

#Preview {
    ContentView()
}
