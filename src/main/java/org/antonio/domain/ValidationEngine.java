package org.antonio.domain;

import org.antonio.Main;
import org.antonio.bridge.DatasetStatTracker;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.perception.FeatureExtractor;
import org.antonio.perception.PcFeatureExtractor;
import org.antonio.vsa.VsaVector;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Motore di Validazione Batch.
 * Testa l'intero dataset di validazione contro gli archetipi pre-calcolati
 * e restituisce l'accuratezza globale e per singola classe.
 */
public class ValidationEngine {

    private final FeatureExtractor cnn;
    private final RandomProjectionBridge bridge;
    private final DatasetStatTracker tracker;
    private final Map<String, VsaVector> archetypes;

    public ValidationEngine(String modelPath, String statsFilePath, String archetypesFilePath) throws Exception {
        System.out.println("\n[VALIDAZIONE] Inizializzazione Motore Edge...");
        this.cnn = new PcFeatureExtractor(modelPath);
        this.bridge = new RandomProjectionBridge(1280, 42L);
        this.tracker = DatasetStatTracker.importFromFile(statsFilePath);
        this.archetypes = loadArchetypes(archetypesFilePath);
    }

    /**
     * Esegue il test su un'intera cartella strutturata (es. dataset_val/Classe/foto.jpg)
     */
    public void runValidation(String validationDirPath) {
        File rootDir = new File(validationDirPath);
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            throw new IllegalArgumentException("Cartella di validazione non trovata: " + validationDirPath);
        }

        File[] classFolders = rootDir.listFiles(File::isDirectory);
        if (classFolders == null || classFolders.length == 0) {
            System.out.println("Nessuna cartella di classe trovata in " + validationDirPath);
            return;
        }

        int totalImages = 0;
        int globalCorrect = 0;
        int globalUnknown = 0; // Immagini scartate (sotto i 3 Sigma)

        // Statistiche per singola classe: [0] = Corrette, [1] = Totali
        Map<String, int[]> classStats = new HashMap<>();

        System.out.println("\n=== INIZIO VALIDAZIONE BATCH ===");

        for (File classFolder : classFolders) {
            String trueClass = classFolder.getName();
            classStats.put(trueClass, new int[]{0, 0});

            File[] images = classFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png"));
            if (images == null || images.length == 0) continue;

            System.out.println("Test in corso sulla classe: [" + trueClass + "] - " + images.length + " immagini...");

            for (File imgFile : images) {
                totalImages++;
                classStats.get(trueClass)[1]++; // Incrementa il totale di questa classe

                try {
                    float[][][] imageTensor = Main.loadImage(imgFile.getAbsolutePath());
                    String predictedClass = predict(imageTensor);

                    if (predictedClass.equals(trueClass)) {
                        globalCorrect++;
                        classStats.get(trueClass)[0]++; // Incrementa corrette per questa classe
                    } else if (predictedClass.equals("Sconosciuta")) {
                        globalUnknown++;
                    }

                } catch (Exception e) {
                    System.err.println("Errore sull'immagine " + imgFile.getName() + ": " + e.getMessage());
                }
            }
        }

        // --- STAMPA DEL REPORT FINALE ---
        System.out.println("\n==================================================");
        System.out.println("            REPORT DI VALIDAZIONE VSA             ");
        System.out.println("==================================================");

        System.out.printf("Totale Immagini Testate : %d\n", totalImages);
        System.out.printf("Predizioni Corrette     : %d\n", globalCorrect);
        System.out.printf("Scartate (Sconosciuta)  : %d (Sotto i 3 Sigma)\n", globalUnknown);
        System.out.printf("Falsi Positivi/Errori   : %d\n", totalImages - globalCorrect - globalUnknown);

        double overallAccuracy = (double) globalCorrect / totalImages * 100.0;
        System.out.printf("\n---> ACCURATEZZA GLOBALE: %.2f%% <---\n", overallAccuracy);

        System.out.println("\n--- Dettaglio per Classe ---");
        for (Map.Entry<String, int[]> entry : classStats.entrySet()) {
            String className = entry.getKey();
            int correct = entry.getValue()[0];
            int total = entry.getValue()[1];
            if (total == 0) continue;

            double accuracy = (double) correct / total * 100.0;
            System.out.printf("[%-40s] : %5.1f%% (%d/%d)\n", className, accuracy, correct, total);
        }
        System.out.println("==================================================");
    }

    /**
     * Esegue l'inferenza pura su una singola immagine.
     */
    private String predict(float[][][] imageTensor) {
        float[] features = cnn.extractFeatures(imageTensor);
        double[] continuous = bridge.projectToContinuous(features);
        double[] standardized = tracker.standardize(continuous);
        VsaVector queryVector = bridge.binarize(standardized);

        String bestMatch = "Sconosciuta";
        double maxSimilarity = -1.0;
        double sigma = 1.0 / Math.sqrt(VsaVector.DIMENSIONS);
        double bestZScore = 0.0;

        for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
            double similarity = queryVector.getCosineSimilarity(entry.getValue());
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestMatch = entry.getKey();
                bestZScore = similarity / sigma;
            }
        }

        // Applichiamo la soglia di sicurezza: se la migliore risposta è sotto i 3 Sigma, la scartiamo.
        if (bestZScore < 3.0) {
            return "Sconosciuta";
        }

        return bestMatch;
    }

    private Map<String, VsaVector> loadArchetypes(String filepath) throws Exception {
        Map<String, VsaVector> loaded = new HashMap<>();
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filepath))) {
            int numClasses = dis.readInt();
            int dimensions = dis.readInt();
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
        return loaded;
    }

    public static void main(String[] args) {
        String modelPath = "src/main/resources/mobilenet_savedmodel";
        String statsFile = "edge_brain/vsa_stats.bin";
        String archetypesFile = "edge_brain/vsa_archetypes.bin";

        // Sostituisci questo con il percorso reale della tua cartella "val" di PlantVillage
        String validationDir = "C:/Users/Ion/IdeaProjects/Vaimee/VSA/plant-angine-main/val";

        try {
            ValidationEngine validator = new ValidationEngine(modelPath, statsFile, archetypesFile);
            validator.runValidation(validationDir);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}