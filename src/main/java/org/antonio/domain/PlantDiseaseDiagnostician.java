package org.antonio.domain;

import org.antonio.bridge.DatasetStatTracker;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.perception.FeatureExtractor;
import org.antonio.vsa.VsaVector;

import java.util.List;

/**
 * Il "Medico" Neuro-Simbolico.
 * Coordina l'estrazione visiva (CNN), la proiezione matematica (Bridge),
 * la standardizzazione (Tracker) e la memoria vettoriale (VSA).
 */
public class PlantDiseaseDiagnostician {

    private final FeatureExtractor cnn;
    private final RandomProjectionBridge bridge;
    private final DatasetStatTracker tracker;

    // Archetipi definitivi, puramente binari (per la massima velocità in inferenza)
    private VsaVector healthyReference;
    private VsaVector sickReference;

    public PlantDiseaseDiagnostician(FeatureExtractor cnn, RandomProjectionBridge bridge, DatasetStatTracker tracker) {
        this.cnn = cnn;
        this.bridge = bridge;
        this.tracker = tracker;
    }

    /**
     * ADDESTRAMENTO: Insegna cos'è una "Pianta Sana" usando un BATCH di immagini.
     * @param images Lista di tensori immagine
     */
    public void teachHealthy(List<float[][][]> images) {
        this.healthyReference = createArchetype(images);
        System.out.println("Archetipo 'Sana' creato tramite Bundling di " + images.size() + " immagini.");
    }

    /**
     * ADDESTRAMENTO: Insegna cos'è la "Malattia" usando un BATCH di immagini.
     * @param images Lista di tensori immagine
     */
    public void teachSick(List<float[][][]> images) {
        this.sickReference = createArchetype(images);
        System.out.println("Archetipo 'Malata' creato tramite Bundling di " + images.size() + " immagini.");
    }

    /**
     * IL CUORE DEL BUNDLING: Late Binarization
     * 1. Estrae e proietta ogni immagine.
     * 2. Le standardizza.
     * 3. Le SOMMA in un unico super-vettore continuo (Bundling).
     * 4. Binarizza il risultato finale.
     */
    private VsaVector createArchetype(List<float[][][]> images) {
        if (images == null || images.isEmpty()) {
            throw new IllegalArgumentException("La lista di immagini per l'addestramento non può essere vuota.");
        }

        // Il "Bundle" continuo che accumulerà l'informazione
        double[] bundle = new double[VsaVector.DIMENSIONS];

        for (float[][][] image : images) {
            // A. Occhio (CNN)
            float[] features = cnn.extractFeatures(image);

            // B. Proiezione Olografica e Normalizzazione L2
            double[] continuous = bridge.projectToContinuous(features);

            // C. Centratura (Z-Score) per garantire l'ortogonalità
            double[] standardized = tracker.standardize(continuous);

            // D. BUNDLING CONTINUO: Somma elemento per elemento
            for (int d = 0; d < VsaVector.DIMENSIONS; d++) {
                bundle[d] += standardized[d];
            }
        }

        // E. LATE BINARIZATION: Il Bundle continuo diventa un Ipervettore BSC puro
        return bridge.binarize(bundle);
    }

    /**
     * INFERENZA: Diagnostica una nuova foto dal campo.
     */
    public String diagnose(float[][][] image) {
        if (healthyReference == null || sickReference == null) {
            throw new IllegalStateException("Devi prima addestrare il sistema con dei batch di immagini!");
        }

        // --- PREPROCESSING DELLA QUERY ---
        float[] features = cnn.extractFeatures(image);
        double[] continuous = bridge.projectToContinuous(features);
        double[] standardized = tracker.standardize(continuous); // Usa le statistiche pre-calcolate!
        VsaVector queryVector = bridge.binarize(standardized);   // Ipervettore binario di test

        // --- CALCOLO DELLA DISTANZA DI HAMMING (Hardware Popcnt) ---
        double simHealthy = queryVector.getCosineSimilarity(healthyReference);
        double simSick = queryVector.getCosineSimilarity(sickReference);

        // --- SOGLIA STATISTICA (Z-Score VSA) ---
        // La deviazione standard teorica di uno spazio BSC casuale è 1 / sqrt(D)
        double sigma = 1.0 / Math.sqrt(VsaVector.DIMENSIONS);
        double zScoreHealthy = simHealthy / sigma;
        double zScoreSick = simSick / sigma;

        System.out.println("   --- ANALISI MATEMATICA VSA ---");
        System.out.printf("   Similarità [Pianta Sana]   : %.4f (Z-Score: %.1f Sigma)%n", simHealthy, zScoreHealthy);
        System.out.printf("   Similarità [Pianta Malata] : %.4f (Z-Score: %.1f Sigma)%n", simSick, zScoreSick);

        // Se l'immagine non assomiglia a nulla (è un gatto, un tavolo, o un'altra malattia)
        if (Math.max(zScoreHealthy, zScoreSick) < 3.0) {
            return "Sconosciuta (Troppo rumore, la rete non è sicura!)";
        }

        return simSick > simHealthy ? "Malata!" : "Sana!";
    }
}