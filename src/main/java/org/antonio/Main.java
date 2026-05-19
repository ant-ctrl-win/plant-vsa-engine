package org.antonio;

import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.domain.PlantDiseaseDiagnostician;
import org.antonio.perception.FeatureExtractor;
import org.antonio.perception.PcFeatureExtractor;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;

public class Main {

    public static void main(String[] args) {
        System.out.println("--- AVVIO MOTORE VSA/HDC SU PC ---");

        // Assicurati che questo percorso punti alla cartella estratta da Colab
        String modelPath = "src/main/resources/mobilenet_savedmodel";
        FeatureExtractor realPcEye = null;

        try {
            // 1. INIZIALIZZAZIONE DEL SISTEMA NEURO-SIMBOLICO
            realPcEye = new PcFeatureExtractor(modelPath);
            RandomProjectionBridge bridge = new RandomProjectionBridge(1280, 42L); // 42 è il nostro seme fisso
            PlantDiseaseDiagnostician doctor = new PlantDiseaseDiagnostician(realPcEye, bridge);

            // 2. CARICAMENTO DELLE FOTO REALI
            // Assicurati di avere queste tre foto (es. .jpg) nella root del tuo progetto IntelliJ
            System.out.println("\nCaricamento immagini dal disco in corso...");
            float[][][] imageHealthy = loadImage("sana.jpg");
            float[][][] imageSick = loadImage("malata.jpg");
            float[][][] imageTest = loadImage("test.jpg");
            System.out.println("Immagini caricate e processate con successo.");

            // 3. FEW-SHOT LEARNING (Apprendimento sul campo)
            System.out.println("\n[Fase 1: Few-Shot Learning]");
            doctor.teachHealthy(imageHealthy);
            doctor.teachSick(imageSick);

            // 4. DIAGNOSI DELL'IMMAGINE INCOGNITA
            System.out.println("\n[Fase 2: Diagnosi della nuova foto]");
            String diagnosis = doctor.diagnose(imageTest);

            System.out.println("\n*** VERDETTO FINALE: " + diagnosis + " ***");

        } catch (Exception e) {
            System.err.println("Errore durante l'esecuzione:");
            e.printStackTrace();
        } finally {
            // 5. PULIZIA DELLA MEMORIA
            if (realPcEye instanceof PcFeatureExtractor) {
                ((PcFeatureExtractor) realPcEye).close();
                System.out.println("\nMotore TensorFlow chiuso correttamente.");
            }
        }
    }

    /**
     * Legge un'immagine dal disco, la ridimensiona a 224x224 e la normalizza
     * nel formato richiesto da MobileNetV2 (valori tra -1.0 e 1.0).
     */
    public static float[][][] loadImage(String imagePath) throws Exception {
        File file = new File(imagePath);
        if (!file.exists()) {
            throw new RuntimeException("Immagine non trovata: " + file.getAbsolutePath());
        }

        BufferedImage originalImg = ImageIO.read(file);

        // Ridimensionamento forzato a 224x224 pixel
        BufferedImage resizedImg = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImg.createGraphics();
        g.drawImage(originalImg, 0, 0, 224, 224, null);
        g.dispose();

        // Estrazione RGB e normalizzazione Keras
        float[][][] tensor = new float[224][224][3];
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int rgb = resizedImg.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF);
                float g_color = ((rgb >> 8) & 0xFF);
                float b = (rgb & 0xFF);

                // Formula di Keras per MobileNetV2: (valore / 127.5) - 1.0
                tensor[y][x][0] = (r / 127.5f) - 1.0f;
                tensor[y][x][1] = (g_color / 127.5f) - 1.0f;
                tensor[y][x][2] = (b / 127.5f) - 1.0f;
            }
        }
        return tensor;
    }
}