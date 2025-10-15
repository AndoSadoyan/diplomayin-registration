package am.mt240.asadoyan.registration;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class RegistrationTerminal {
    private JFrame frame;
    private volatile boolean running = false;
    private volatile boolean faceDetected;
    private volatile long faceDetectionTime;
    private final CascadeClassifier faceDetector;
    private OpenCVFrameGrabber grabber;
    private final OrtEnvironment env;
    private final OrtSession session;
    ExecutorService executor;

    public RegistrationTerminal() {
        try {
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(getClass().getClassLoader()
                    .getResource("models/arcfaceresnet100-insightface.onnx").getPath(), new OrtSession.SessionOptions());
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        faceDetector = new CascadeClassifier(getClass().getClassLoader()
                .getResource("haarcascade_frontalface_default.xml").getPath());

        initIdInputScene();
    }

    private void initIdInputScene() {
        frame = new JFrame("Student Registration");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 150);
        frame.setLocationRelativeTo(null);

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new FlowLayout());

        JTextField idField = new JTextField(20);
        JButton okButton = new JButton("OK");

        mainPanel.add(new JLabel("Enter Student ID:"));
        mainPanel.add(idField);
        mainPanel.add(okButton);

        okButton.addActionListener(e -> {
            String studentId = idField.getText().trim();
            if (!studentId.isEmpty()) {
                frame.getContentPane().removeAll();
                frame.repaint();
                startFaceCaptureScene(studentId);
            }
        });

        frame.setContentPane(mainPanel);
        frame.setVisible(true);
    }

    private void startFaceCaptureScene(String studentId) {
        JPanel capturePanel = new JPanel(new BorderLayout());

        CameraPanel cameraPanel = new CameraPanel();
        cameraPanel.setPreferredSize(new Dimension(640, 480));
        capturePanel.add(cameraPanel, BorderLayout.CENTER);

        JButton okBtn = new JButton("OK");
        JButton retryBtn = new JButton("Retry");
        JButton cancelBtn = new JButton("Cancel");

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(okBtn);
        buttonPanel.add(retryBtn);
        buttonPanel.add(cancelBtn);
        capturePanel.add(buttonPanel, BorderLayout.SOUTH);

        frame.setContentPane(capturePanel);
        frame.setSize(550, 600);
        frame.setLocationRelativeTo(null);
        frame.revalidate();

        grabber = new OpenCVFrameGrabber(0);
        try {
            grabber.start();
        } catch (FrameGrabber.Exception ex) {
            ex.printStackTrace();
            return;
        }

        executor = Executors.newSingleThreadExecutor();
        CompletableFuture<Mat> capturedFaceFuture = new CompletableFuture<>();

        running = true;
        faceDetected = false;
        faceDetectionTime = 0;

        executor.submit(() -> {
            try {
                boolean captured = false;
                while (running && !captured) {
                    org.bytedeco.javacv.Frame frameGrab = grabber.grab();
                    if (frameGrab == null) continue;

                    Mat mat = new OpenCVFrameConverter.ToMat().convert(frameGrab);
                    // Convert to grayscale
                    Mat gray = new Mat();
                    cvtColor(mat, gray, COLOR_BGR2GRAY);

                    // Detect faces in center region
                    RectVector faces = new RectVector();
                    faceDetector.detectMultiScale(gray, faces);

                    if (faces.size() > 0) {
                        Rect closestFace = null;
                        int maxArea = 0;

                        for (int i = 0; i < faces.size(); i++) {
                            Rect faceRect = faces.get(i);
                            int area = faceRect.width() * faceRect.height();
                            if (area > maxArea && faceRect.height() > 200) {
                                maxArea = area;
                                closestFace = faceRect;
                            }
                        }

                        if (closestFace != null) {
                            // Adjust coordinates to full frame
                            Rect adjustedRect = safeRect(closestFace, mat);

                            // Draw rectangle around detected face
                            rectangle(mat, adjustedRect, new Scalar(0, 255, 0, 255));

                            if (!faceDetected) {
                                faceDetected = true;
                                faceDetectionTime = System.currentTimeMillis();
                            }

                            // If face is stable for 2s -> capture it
                            if (faceDetected && (System.currentTimeMillis() - faceDetectionTime) >= 2000) {
                                Mat face = new Mat(mat, adjustedRect).clone(); // clone so it’s not overwritten
                                capturedFaceFuture.complete(face);
                                // Show captured face instead of live feed
                                BufferedImage faceImg = matToBufferedImage(face);
                                SwingUtilities.invokeLater(() -> cameraPanel.setImage(faceImg));
                                captured = true; // stop live feed
                            }
                        } else {
                            faceDetected = false;
                            faceDetectionTime = 0;
                        }
                    } else {
                        faceDetected = false;
                        faceDetectionTime = 0;
                    }

                    if (!captured) {
                        // Still live feed
                        int feedCropWidth = (int) (mat.cols() * 0.6); // crop 60% of width
                        int feedCropHeight = mat.rows();             // keep full height
                        int feedX = Math.min((mat.cols() - feedCropWidth)/2, mat.cols() - feedCropWidth);
                        int feedY = 0;

                        Rect displayROI = new Rect(feedX, feedY, feedCropWidth, feedCropHeight);
                        Mat displayMat = new Mat(mat, displayROI);

                        // Convert to BufferedImage and show
                        BufferedImage frameImg = matToBufferedImage(displayMat);
                        SwingUtilities.invokeLater(() -> cameraPanel.setImage(frameImg));
                    }

                    Thread.sleep(33); // ~30fps
                }
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });

        // --- Button actions ---
        okBtn.addActionListener(e -> {
            if (capturedFaceFuture.isDone()) {
                Mat faceMat = capturedFaceFuture.join();
                float[] embedding = computeEmbedding(faceMat);
                sendEmbeddingToBackend(studentId, embedding);
                stopGrabber();
                running = false;
                frame.dispose();
                initIdInputScene();
            }
        });

        retryBtn.addActionListener(e -> {
            stopGrabber();
            running = false;
            // Don’t dispose the frame, just replace its content
            startFaceCaptureScene(studentId);
        });

        cancelBtn.addActionListener(e -> {
            stopGrabber();
            running = false;
            frame.dispose();
            initIdInputScene();
        });
    }

    private Rect safeRect(Rect rect, Mat mat) {
        int x1 = Math.max(rect.x() - 10, 0);
        int y1 = Math.max(rect.y() - 10, 0);
        int x2 = Math.min(rect.x() + rect.width() + 10, mat.cols());
        int y2 = Math.min(rect.y() + rect.height() + 10, mat.rows());

        int w = Math.max(x2 - x1, 1);
        int h = Math.max(y2 - y1, 1);

        return new Rect(x1, y1, w, h);
    }


    private void stopGrabber() {
        executor.shutdownNow();
        try {
            if (grabber != null) {
                grabber.stop();
                grabber.release();
            }
        } catch (FrameGrabber.Exception e) {
            throw new RuntimeException(e);
        }
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter java2DConverter = new Java2DFrameConverter();
        org.bytedeco.javacv.Frame frame = converter.convert(mat);
        return java2DConverter.getBufferedImage(frame, 1);
    }

    private float[] matToCHWFloatArray(Mat mat) {
        int channels = mat.channels(); // should be 3
        int width = mat.cols();
        int height = mat.rows();

        float[] chw = new float[channels * width * height];
        FloatIndexer indexer = mat.createIndexer();

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float val = indexer.get(y, x, c);
                    chw[c * height * width + y * width + x] = val;
                }
            }
        }

        indexer.release();
        return chw;
    }

    public float[] computeEmbedding(Mat faceMat) {
        try {
            // 1. Resize to 112x112 - KEEP BGR (don't convert to RGB)
            Mat resized = new Mat();
            resize(faceMat, resized, new Size(112, 112));
            // 2. Convert to float32 and normalize using mean-std (typical for deep learning)
            resized.convertTo(resized, CV_32F);
            // Subtract mean [127.5, 127.5, 127.5] and divide by 127.5 (range: -1 to 1)
            resized = subtract(resized, new Scalar(127.5, 127.5, 127.5, 0.0)).asMat();
            resized = multiply(resized, 1.0 / 127.5).asMat();
            // 3. Convert to CHW float array
            float[] chw = matToCHWFloatArray(resized);
            // 5. Create input tensor
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, 3, 112, 112});
            // 6. Run inference (InsightFace model uses "input.1" instead of "data")
            OrtSession.Result result = session.run(Collections.singletonMap("input.1", inputTensor));
            float[][] output = (float[][]) result.get(0).getValue();

            float[] normalized = normalize(output[0]); // 512-d embedding
            System.out.println("Embedding generated: " + Arrays.toString(normalized));
            return normalized;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }


    public float[] normalize(float[] embedding) {
        float norm = 0f;
        for (float v : embedding) norm += v * v;
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < embedding.length; i++) embedding[i] /= norm;
        return embedding;
    }

    private void sendEmbeddingToBackend(String studentId, float[] embedding) {
        // Placeholder: implement API call to backend
        System.out.println("Sending embedding for student " + studentId);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(RegistrationTerminal::new);
    }
}