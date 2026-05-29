package org.antonio.domain;

import org.antonio.Main;
import org.antonio.bridge.DatasetStatTracker;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.perception.FeatureExtractor;
import org.antonio.perception.PcFeatureExtractor;
import org.antonio.vsa.VsaVector;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Simula l'App Android o il dispositivo Edge.
 * Carica solo i file binari pre-calcolati (NO Addestramento).
 */
public class EdgeInferenceEngine {

    private final FeatureExtractor cnn;
    private final RandomProjectionBridge bridge;
    private final DatasetStatTracker tracker;
    private final Map<String, VsaVector> archetypes;

    /**
     * Inizializza il motore caricando il "cervello" pre-calcolato
     */
    public EdgeInferenceEngine(String modelPath, String statsFilePath, String archetypesFilePath) throws Exception {
        System.out.println("\n[EDGE] Avvio Motore di Inferenza Leggero...");

        // 1. Inizializza l'occhio CNN (MobileNetV2)
        this.cnn = new PcFeatureExtractor(modelPath);

        // 2. Inizializza la matrice logica fissa
        this.bridge = new RandomProjectionBridge(FeatureExtractor.CNN_FEATURE_SIZE, 42L);

        // 3. Carica le medie e le deviazioni standard globali dal file (Frazione di millisecondo)
        this.tracker = DatasetStatTracker.importFromFile(statsFilePath);

        // 4. Carica le malattie (Archetipi)
        this.archetypes = loadArchetypes(archetypesFilePath);
    }

    /**
     * Effettua la diagnosi di una nuova immagine in input
     */
    public void diagnose(float[][][] imageTensor) {
        System.out.println("\n[EDGE] Inizio Inferenza...");
        long startTime = System.currentTimeMillis();

        // 1. Occhio (Estrazione CNN)
        float[] features = cnn.extractFeatures(imageTensor);
        long cnnTime = System.currentTimeMillis();

        // 2. Matematica (Proiezione + Z-Score + Binarizzazione)
        double[] continuous = bridge.projectToContinuous(features);
        double[] standardized = tracker.standardize(continuous);
        VsaVector queryVector = bridge.binarize(standardized);
        long mathTime = System.currentTimeMillis();

        // 3. Ricerca Distanze e Voto
        String bestMatch = "Sconosciuta";
        double maxSimilarity = -1.0;

        System.out.println("\n--- ANALISI VSA (Similarità del Coseno) ---");

        for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
            String className = entry.getKey();
            VsaVector archetype = entry.getValue();

            // L'hardware calcola lo XOR e conta i bit
            double similarity = queryVector.getCosineSimilarity(archetype);

            // Calcolo Z-Score per mostrare la confidenza
            double sigma = 1.0 / Math.sqrt(VsaVector.DIMENSIONS);
            double zScore = similarity / sigma;

            System.out.printf("   [%-40s] : %.4f (Z-Score: %5.1f Sigma)\n", className, similarity, zScore);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestMatch = className;
            }
        }

        long endTime = System.currentTimeMillis();

        System.out.println("\n*** VERDETTO: " + bestMatch + " ***");
        System.out.println("\n--- TEMPI DI ESECUZIONE ---");
        System.out.println("CNN Feature Extraction: " + (cnnTime - startTime) + " ms");
        System.out.println("Matematica VSA (Edge):  " + (mathTime - cnnTime) + " ms");
        System.out.println("Confronto Archetipi:    " + (endTime - mathTime) + " ms");
        System.out.println("Totale Inferenza:       " + (endTime - startTime) + " ms");
    }

    /**
     * Legge il file binario compatto e ricostruisce i VsaVector in RAM
     */
    private Map<String, VsaVector> loadArchetypes(String filepath) throws Exception {
        Map<String, VsaVector> loaded = new HashMap<>();
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filepath))) {
            int numClasses = dis.readInt();
            int dimensions = dis.readInt();

            if (dimensions != VsaVector.DIMENSIONS) {
                throw new IllegalStateException("Dimensionalità non corrispondente!");
            }

            for (int i = 0; i < numClasses; i++) {
                String className = dis.readUTF();
                int numBlocks = dis.readInt();
                long[] blocks = new long[numBlocks];

                for (int j = 0; j < numBlocks; j++) {
                    blocks[j] = dis.readLong();
                }

                loaded.put(className, new VsaVector(blocks));
            }
        }
        System.out.println("Caricati " + loaded.size() + " archetipi da " + filepath);
        return loaded;
    }

    // Metodo main per testare lo script
    public static void main(String[] args) {
        String modelPath = "src/main/resources/mobilenet_savedmodel";
        String statsFile = "edge_brain/vsa_stats.bin";
        String archetypesFile = "edge_brain/vsa_archetypes.bin";
        String testImage = "vite_test.jpg"; // <-- Assicurati di avere un'immagine di test qui

        try {
            EdgeInferenceEngine edgeDevice = new EdgeInferenceEngine(modelPath, statsFile, archetypesFile);

            // Usiamo il caricatore del tuo Main principale
            float[][][] imageTensor = Main.loadImage(testImage);

            edgeDevice.diagnose(imageTensor);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}