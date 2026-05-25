package org.antonio.domain;

import org.antonio.Main;
import org.antonio.bridge.DatasetStatTracker;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.perception.FeatureExtractor;
import org.antonio.vsa.VsaVector;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Lo script di automazione per addestrare l'intero dataset PlantVillage.
 * Estrae le feature, calcola lo Z-Score globale, esegue il bundling per ogni classe
 * ed esporta il "cervello" in file binari leggerissimi per l'Edge Computing.
 */
public class PlantVillageTrainer {

    private final FeatureExtractor cnn;
    private final RandomProjectionBridge bridge;
    private final DatasetStatTracker tracker;

    public PlantVillageTrainer(FeatureExtractor cnn, RandomProjectionBridge bridge, DatasetStatTracker tracker) {
        this.cnn = cnn;
        this.bridge = bridge;
        this.tracker = tracker;
    }

    /**
     * Esegue l'intero processo di addestramento su una cartella strutturata.
     * Struttura attesa: datasetPath / NomeMalattia / immagine.jpg
     */
    public void trainDataset(String datasetPath, String outputDir) throws Exception {
        File rootDir = new File(datasetPath);
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            throw new IllegalArgumentException("Cartella dataset non trovata: " + datasetPath);
        }

        // Cache in RAM per evitare di ripassare le immagini nella CNN due volte
        Map<String, List<float[]>> extractedFeaturesByClass = new HashMap<>();
        long totalImages = 0;

        System.out.println("\n=== FASE 1: Estrazione Feature CNN (Operazione Pesante) ===");

        File[] classFolders = rootDir.listFiles(File::isDirectory);
        if (classFolders == null) throw new IOException("Nessuna cartella di classe trovata.");

        for (File classFolder : classFolders) {
            String className = classFolder.getName();
            File[] images = classFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".png"));

            if (images == null || images.length == 0) continue;

            System.out.println("Elaborazione classe [" + className + "] - " + images.length + " immagini...");
            List<float[]> featuresList = new ArrayList<>();

            for (File imgFile : images) {
                try {
                    // Usa il metodo che abbiamo già in Main.java
                    float[][][] tensor = Main.loadImage(imgFile.getAbsolutePath());
                    float[] features = cnn.extractFeatures(tensor);
                    featuresList.add(features);
                    totalImages++;
                } catch (Exception e) {
                    System.err.println("Errore caricamento " + imgFile.getName() + " - " + e.getMessage());
                }
            }
            extractedFeaturesByClass.put(className, featuresList);
        }

        System.out.println("Estrazione completata! Totale immagini elaborate: " + totalImages);

        System.out.println("\n=== FASE 2: Calcolo Statistiche Globali (Z-Score) ===");
        // Passiamo tutte le feature continue al tracker
        for (List<float[]> classFeatures : extractedFeaturesByClass.values()) {
            for (float[] features : classFeatures) {
                double[] continuous = bridge.projectToContinuous(features);
                tracker.observe(continuous);
            }
        }
        tracker.finalizeStats();

        // Salviamo le statistiche (Mu e Sigma)
        new File(outputDir).mkdirs();
        tracker.exportToFile(outputDir + "/vsa_stats.bin");

        System.out.println("\n=== FASE 3: Bundling e Binarizzazione (Late Binarization) ===");
        Map<String, VsaVector> finalArchetypes = new HashMap<>();

        for (Map.Entry<String, List<float[]>> entry : extractedFeaturesByClass.entrySet()) {
            String className = entry.getKey();
            List<float[]> featuresList = entry.getValue();

            double[] bundle = new double[VsaVector.DIMENSIONS];

            for (float[] features : featuresList) {
                double[] continuous = bridge.projectToContinuous(features);
                double[] standardized = tracker.standardize(continuous); // Z-Score globale applicato!

                // Accumulo nel bundle continuo
                for (int d = 0; d < VsaVector.DIMENSIONS; d++) {
                    bundle[d] += standardized[d];
                }
            }

            // Binarizzazione finale del super-vettore
            VsaVector archetype = bridge.binarize(bundle);
            finalArchetypes.put(className, archetype);
            System.out.println("Archetipo creato per: " + className);
        }

        System.out.println("\n=== FASE 4: Esportazione della Conoscenza ===");
        exportArchetypes(finalArchetypes, outputDir + "/vsa_archetypes.bin");
        System.out.println("Addestramento completato! Il tuo sistema è pronto per l'Edge.");
    }

    /**
     * Salva tutti gli archetipi in un file binario ottimizzato per Android/Microcontrollori.
     */
    private void exportArchetypes(Map<String, VsaVector> archetypes, String filepath) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filepath))) {
            dos.writeInt(archetypes.size()); // Quante malattie ci sono?
            dos.writeInt(VsaVector.DIMENSIONS); // Quante dimensioni VSA?

            for (Map.Entry<String, VsaVector> entry : archetypes.entrySet()) {
                dos.writeUTF(entry.getKey()); // Scrive il nome della malattia

                long[] blocks = entry.getValue().getBlocks();
                dos.writeInt(blocks.length); // Scrive quanti "long" compongono l'array (157)

                for (long block : blocks) {
                    dos.writeLong(block); // Scrive i bit effettivi
                }
            }
        }
        System.out.println("Archetipi salvati in: " + filepath);
    }
}