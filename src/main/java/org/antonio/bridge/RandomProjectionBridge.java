package org.antonio.bridge;
import org.antonio.vsa.VsaVector;

import java.util.Random;

/**
 * Traduce i float in uscita dalla CNN in bit per la VSA (SBC).
 */
public class RandomProjectionBridge {

    private final int cnnFeatureSize;
    private final int vsaDimensions;

    // Invece di float, usiamo boolean per la matrice per risparmiare RAM!
    // true = +1, false = -1
    private final boolean[][] projectionMatrix;

    /**
     * Inizializza il ponte.
     * @param cnnFeatureSize Es. 512 (Test B) o 38 (Test A)
     * @param seed Il seme fisso per garantire che la matrice sia sempre identica
     */
    public RandomProjectionBridge(int cnnFeatureSize, long seed) {
        this.cnnFeatureSize = cnnFeatureSize;
        this.vsaDimensions = VsaVector.DIMENSIONS;
        this.projectionMatrix = new boolean[cnnFeatureSize][vsaDimensions];

        // Inizializziamo il generatore con il nostro seme fisso
        Random random = new Random(seed);

        // Riempiamo la matrice di +1 (true) e -1 (false)
        for (int i = 0; i < cnnFeatureSize; i++) {
            for (int d = 0; d < vsaDimensions; d++) {
                projectionMatrix[i][d] = random.nextBoolean();
            }
        }
    }

    /**
     * Il momento magico: da array continuo (CNN) a Ipervettore BSC
     */
    public VsaVector projectToVsa(float[] cnnFeatures) {
        if (cnnFeatures.length != cnnFeatureSize) {
            throw new IllegalArgumentException("Le feature CNN non combaciano con la matrice!");
        }

        // Calcoliamo quanti 'long' ci servono per contenere 10000 bit (157)
        int numBlocks = (vsaDimensions + 63) / 64;
        long[] bscBlocks = new long[numBlocks];

        // Iteriamo su ogni dimensione del nuovo ipervettore (da 0 a 9999)
        for (int d = 0; d < vsaDimensions; d++) {

            // 1. Calcoliamo il Prodotto Scalare (Dot Product)
            float sum = 0.0f;
            for (int i = 0; i < cnnFeatureSize; i++) {
                // Se la matrice in quella cella è true (+1), sommiamo la feature.
                // Se è false (-1), la sottraiamo.
                if (projectionMatrix[i][d]) {
                    sum += cnnFeatures[i];
                } else {
                    sum -= cnnFeatures[i];
                }
            }

            // 2. La Funzione Segno (Thresholding)
            // Se la somma è > 0, decidiamo che il bit sarà 1. Se <= 0, sarà 0.
            if (sum > 0) {
                // 3. Bit-Packing: Inseriamo l'1 nel posto giusto dentro l'array di long!
                int blockIndex = d / 64; // Trova in quale dei 157 long dobbiamo andare
                int bitPosition = d % 64; // Trova quale dei 64 bit di QUEL long dobbiamo accendere

                // Usiamo l'operatore di Shift (<<) e l'operatore OR (|)
                bscBlocks[blockIndex] |= (1L << bitPosition);
            }
            // Se sum <= 0, non facciamo nulla, perché gli array in Java nascono già pieni di zeri.
        }

        return new VsaVector(bscBlocks);
    }
}