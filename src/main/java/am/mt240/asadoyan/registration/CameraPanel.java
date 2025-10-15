package am.mt240.asadoyan.registration;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class CameraPanel extends JPanel {
    private BufferedImage image;

    public void setImage(BufferedImage img) {
        this.image = img;
        repaint();
    }

    @Override

    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (image != null) {
            // Get panel size
            int panelWidth = getWidth();
            int panelHeight = getHeight();

            // Get image size
            int imgWidth = image.getWidth();
            int imgHeight = image.getHeight();

            // Calculate scale to fit image within panel while preserving aspect ratio
            double scale = Math.min((double) panelWidth / imgWidth, (double) panelHeight / imgHeight);
            int drawWidth = (int) (imgWidth * scale);
            int drawHeight = (int) (imgHeight * scale);

            // Center the image
            int x = (panelWidth - drawWidth) / 2;
            int y = (panelHeight - drawHeight) / 2;

            g.drawImage(image, x, y, drawWidth, drawHeight, null);
        }
    }

}