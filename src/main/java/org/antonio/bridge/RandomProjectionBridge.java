package org.antonio.bridge;

import org.antonio.vsa.VsaVector;
import java.util.Random;

/**
 * Traduce i float in uscita dalla CNN in vettori per la VSA (SBC).
 * Implementa il paradigma "Late Binarization" e la Normalizzazione L2.
 */
public class RandomProjectionBridge {

    private final int cnnFeatureSize;
    private final int vsaDimensions;

    // Matrice logica per la proiezione casuale (true = +1, false = -1)
    private final boolean[][] projectionMatrix;

    /**
     * Inizializza il ponte con una matrice fissa e deterministica.
     * @param cnnFeatureSize Dimensione output CNN (es. 1280)
     * @param seed Il seme per garantire che la matrice sia sempre identica (es. 42L)
     */
    public RandomProjectionBridge(int cnnFeatureSize, long seed) {
        this.cnnFeatureSize = cnnFeatureSize;
        this.vsaDimensions = VsaVector.DIMENSIONS;
        this.projectionMatrix = new boolean[cnnFeatureSize][vsaDimensions];

        // Inizializziamo il generatore con il seme fisso
        Random random = new Random(seed);

        for (int i = 0; i < cnnFeatureSize; i++) {
            for (int d = 0; d < vsaDimensions; d++) {
                projectionMatrix[i][d] = random.nextBoolean();
            }
        }
    }

    /**
     * FASE 1: Proiezione Continua & Normalizzazione L2
     * Prende le feature della CNN e le spalma su 10.000 dimensioni SENZA binarizzarle.
     * @param cnnFeatures Vettore estratto dalla CNN (es. float[1280])
     * @return Vettore VSA continuo normalizzato (double[10000])
     */
    public double[] projectToContinuous(float[] cnnFeatures) {
        if (cnnFeatures.length != cnnFeatureSize) {
            throw new IllegalArgumentException("Le feature CNN non combaciano con la matrice!");
        }

        double[] continuousVector = new double[vsaDimensions];
        double sumOfSquares = 0.0;

        // 1. Proiezione Casuale (Prodotto Scalare)
        for (int d = 0; d < vsaDimensions; d++) {
            double sum = 0.0;
            for (int i = 0; i < cnnFeatureSize; i++) {
                if (projectionMatrix[i][d]) {
                    sum += cnnFeatures[i];
                } else {
                    sum -= cnnFeatures[i];
                }
            }
            continuousVector[d] = sum;
            sumOfSquares += sum * sum;
        }

        // 2. Normalizzazione L2 (Schiaccia il vettore sulla superficie di un'ipersfera di raggio 1)
        double magnitude = Math.sqrt(sumOfSquares);

        // Evitiamo divisioni per zero se la CNN restituisce un vettore nullo
        if (magnitude > 1e-9) {
            for (int d = 0; d < vsaDimensions; d++) {
                continuousVector[d] /= magnitude;
            }
        }

        return continuousVector;
    }

    /**
     * FASE 2: Late Binarization (Ritorno alla BSC)
     * Trasforma un vettore continuo e standardizzato nell'ipervettore binario finale.
     * @param continuousVector Il vettore (double[10000]) post Z-Score o post Bundling
     * @return VsaVector pronto per il calcolo della Distanza di Hamming
     */
    public VsaVector binarize(double[] continuousVector) {
        if (continuousVector.length != vsaDimensions) {
            throw new IllegalArgumentException("Dimensione vettore continuo errata!");
        }

        int numBlocks = (vsaDimensions + 63) / 64;
        long[] bscBlocks = new long[numBlocks];

        for (int d = 0; d < vsaDimensions; d++) {
            // Funzione Segno: se > 0, il bit diventa 1. Se <= 0, rimane 0.
            if (continuousVector[d] > 0) {
                int blockIndex = d / 64;
                int bitPosition = d % 64;
                bscBlocks[blockIndex] |= (1L << bitPosition);
            }
        }

        return new VsaVector(bscBlocks);
    }
}