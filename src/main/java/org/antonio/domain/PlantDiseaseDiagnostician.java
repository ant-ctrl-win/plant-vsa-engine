package org.antonio.domain;

import org.antonio.vsa.VsaVector;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.perception.FeatureExtractor;

public class PlantDiseaseDiagnostician {

    private final FeatureExtractor cnn;
    private final RandomProjectionBridge bridge;
    private final VsaVector healthyReference; // Vettore memoria "Pianta Sana"
    private final VsaVector sickReference;    // Vettore memoria "Malattia X"

    public PlantDiseaseDiagnostician(FeatureExtractor cnn, RandomProjectionBridge bridge) {
        this.cnn = cnn;
        this.bridge = bridge;
        // In un sistema vero, questi vettori si "imparano" (Few-Shot)
        this.healthyReference = new VsaVector();
        this.sickReference = new VsaVector();
    }

    public String diagnose(float[][][] image) {
        // 1. La percezione "vede" l'immagine
        float[] features = cnn.extractFeatures(image);

        // 2. Il ponte traduce nel linguaggio del ragionamento
        VsaVector queryVector = bridge.projectToVsa(features);

        // 3. Il ragionamento calcola lo Z-Score/Similarità
        double similarityToHealthy = queryVector.calculateCosineSimilarity(healthyReference);
        double similarityToSick = queryVector.calculateCosineSimilarity(sickReference);

        return similarityToSick > similarityToHealthy ? "Malata!" : "Sana!";
    }
}
