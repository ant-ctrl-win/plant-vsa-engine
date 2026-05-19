package org.antonio.domain;

import org.antonio.vsa.VsaVector;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.perception.FeatureExtractor;

public class PlantDiseaseDiagnostician {

    private final FeatureExtractor cnn;
    private final RandomProjectionBridge bridge;

    // Rimuoviamo il "final" perché questi vettori verranno
    // sovrascritti (imparati) durante il Few-Shot Learning!
    private VsaVector healthyReference;
    private VsaVector sickReference;

    public PlantDiseaseDiagnostician(FeatureExtractor cnn, RandomProjectionBridge bridge) {
        this.cnn = cnn;
        this.bridge = bridge;
        // All'avvio, il sistema è una "tabula rasa", non conosce ancora le piante
        this.healthyReference = null;
        this.sickReference = null;
    }

    /**
     * FEW-SHOT LEARNING: Insegna al sistema cos'è una "Pianta Sana"
     * L'utente scatta una foto di una foglia sana e la passa qui.
     */
    public void teachHealthy(float[][][] image) {
        float[] features = cnn.extractFeatures(image);
        this.healthyReference = bridge.projectToVsa(features);
        System.out.println("Vettore 'Sana' memorizzato con successo!");
    }

    /**
     * FEW-SHOT LEARNING: Insegna al sistema cos'è la "Malattia"
     * L'utente scatta una foto di una foglia malata e la passa qui.
     */
    public void teachSick(float[][][] image) {
        float[] features = cnn.extractFeatures(image);
        this.sickReference = bridge.projectToVsa(features);
        System.out.println("Vettore 'Malata' memorizzato con successo!");
    }

    public String diagnose(float[][][] image) {
        if (healthyReference == null || sickReference == null) {
            throw new IllegalStateException("Devi prima addestrare il sistema!");
        }

        float[] features = cnn.extractFeatures(image);
        VsaVector queryVector = bridge.projectToVsa(features);

        double simHealthy = queryVector.getCosineSimilarity(healthyReference);
        double simSick = queryVector.getCosineSimilarity(sickReference);

        // Calcolo dello Z-Score
        double sigma = 1.0 / Math.sqrt(VsaVector.DIMENSIONS); // 0.01 per D=10000
        double zScoreHealthy = simHealthy / sigma;
        double zScoreSick = simSick / sigma;

        System.out.println("   --- ANALISI MATEMATICA VSA ---");
        System.out.printf("   Distanza da [Pianta Sana]   : %.4f (Z-Score: %.1f Sigma)%n", simHealthy, zScoreHealthy);
        System.out.printf("   Distanza da [Pianta Malata] : %.4f (Z-Score: %.1f Sigma)%n", simSick, zScoreSick);

        // Definiamo una soglia minima di certezza (es. 3 Sigma, ~99.7% di confidenza)
        if (Math.max(zScoreHealthy, zScoreSick) < 3.0) {
            return "Sconosciuta (Troppo rumore, la rete non è sicura!)";
        }

        return simSick > simHealthy ? "Malata!" : "Sana!";
    }
}