package org.antonio;

import org.antonio.bridge.DatasetStatTracker;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.domain.PlantDiseaseDiagnostician;
import org.antonio.perception.FeatureExtractor;
import org.antonio.perception.PcFeatureExtractor;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;

public class Main {

    public static void main(String[] args) {
        System.out.println("--- AVVIO MOTORE VSA/HDC SU PC (LATE BINARIZATION) ---");

        // Assicurati che questo percorso punti alla cartella estratta da Colab
        String modelPath = "src/main/resources/mobilenet_savedmodel";
        FeatureExtractor realPcEye = null;

        try {
            // 1. INIZIALIZZAZIONE DEL SISTEMA NEURO-SIMBOLICO
            realPcEye = new PcFeatureExtractor(modelPath);

            // Inizializziamo il Bridge con 1280 feature in input e seme 42
            RandomProjectionBridge bridge = new RandomProjectionBridge(1280, 42L);

            // Inizializziamo il Tracker per le 10.000 dimensioni VSA
            DatasetStatTracker tracker = new DatasetStatTracker(10000);

            // Creiamo il nostro "Medico"
            PlantDiseaseDiagnostician doctor = new PlantDiseaseDiagnostician(realPcEye, bridge, tracker);

            // 2. CARICAMENTO DELLE FOTO REALI
            System.out.println("\nCaricamento immagini dal disco in corso...");
            float[][][] imgH1 = loadImage("sana.jpg");
            float[][][] imgS1 = loadImage("malata.jpg");
            float[][][] imgS2 = loadImage("malata_1.jpg"); // Seconda foto malata
            float[][][] imageTest = loadImage("test.jpg");
            System.out.println("Immagini caricate e processate con successo.");

            // Creiamo i batch (Liste) per l'addestramento
            // Usa singletonList per un singolo elemento (non fa unboxing)
            List<float[][][]> datasetSano = Collections.singletonList(imgH1);

            // Usa asList per più elementi
            List<float[][][]> datasetMalato = Arrays.asList(imgS1, imgS2);

            // 3. FASE DI FIT DELLE STATISTICHE (Z-Score)
            System.out.println("\n[Fase 0: Analisi Statistica del Dataset]");
            // Estraiamo e proiettiamo le feature per farle "osservare" al tracker
            tracker.observe(bridge.projectToContinuous(realPcEye.extractFeatures(imgH1)));
            tracker.observe(bridge.projectToContinuous(realPcEye.extractFeatures(imgS1)));
            tracker.observe(bridge.projectToContinuous(realPcEye.extractFeatures(imgS2)));

            // Chiudiamo i calcoli e generiamo gli array Mu e Sigma
            tracker.finalizeStats();

            // Salviamo il "cervello" statistico su file (Questo file ~80KB andrà su Android!)
            tracker.exportToFile("vsa_stats.bin");

            // 4. CREAZIONE DEGLI ARCHETIPI (Bundling Continuo & Binarizzazione)
            System.out.println("\n[Fase 1: Creazione Archetipi (Bundling)]");
            doctor.teachHealthy(datasetSano);
            doctor.teachSick(datasetMalato);

            // 5. DIAGNOSI DELL'IMMAGINE INCOGNITA
            System.out.println("\n[Fase 2: Diagnosi della nuova foto]");
            String diagnosis = doctor.diagnose(imageTest);

            System.out.println("\n*** VERDETTO FINALE: " + diagnosis + " ***");

        } catch (Exception e) {
            System.err.println("Errore durante l'esecuzione:");
            e.printStackTrace();
        } finally {
            // 6. PULIZIA DELLA MEMORIA
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