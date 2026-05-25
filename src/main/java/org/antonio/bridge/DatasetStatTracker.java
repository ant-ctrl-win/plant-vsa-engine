package org.antonio.bridge;

import java.io.*;

/**
 * Calcola e applica le statistiche globali (Media e Deviazione Standard)
 * necessarie per la Z-Score Standardization dei vettori continui.
 * Implementa l'algoritmo online di Welford per non saturare la RAM.
 */
public class DatasetStatTracker {

    private final int dimensions;
    private long count;

    // Array per le statistiche
    private final double[] mean;
    private final double[] M2; // Somma dei quadrati delle differenze (serve per la varianza)
    private double[] stdDev;

    private boolean isFitComplete = false;

    /**
     * Costruttore per la fase di ADDESTRAMENTO (su PC)
     */
    public DatasetStatTracker(int dimensions) {
        this.dimensions = dimensions;
        this.count = 0;
        this.mean = new double[dimensions];
        this.M2 = new double[dimensions];
    }

    /**
     * Costruttore privato usato internamente per l'INFERENZA (Edge/Android)
     */
    private DatasetStatTracker(int dimensions, double[] mean, double[] stdDev) {
        this.dimensions = dimensions;
        this.mean = mean;
        this.stdDev = stdDev;
        this.M2 = null; // Non serve in inferenza
        this.isFitComplete = true;
    }

    /**
     * FASE 1: OSSERVAZIONE (Solo PC)
     * Passa qui ogni vettore proiettato. Aggiorna media e varianza in tempo reale.
     */
    public void observe(double[] continuousVector) {
        if (continuousVector.length != dimensions) {
            throw new IllegalArgumentException("Il vettore deve avere " + dimensions + " dimensioni.");
        }

        count++;
        for (int i = 0; i < dimensions; i++) {
            double value = continuousVector[i];
            double delta = value - mean[i];

            mean[i] += delta / count;

            double delta2 = value - mean[i];
            M2[i] += delta * delta2;
        }
    }

    /**
     * FASE 2: CHIUSURA DELL'ADDESTRAMENTO (Solo PC)
     * Calcola la deviazione standard finale dopo aver osservato tutto il dataset.
     */
    public void finalizeStats() {
        if (count < 2) {
            throw new IllegalStateException("Servono almeno 2 campioni per calcolare la deviazione standard.");
        }

        stdDev = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            // Varianza campionaria = M2 / (count - 1)
            double variance = M2[i] / (count - 1);

            // Evitiamo divisioni per zero se una dimensione è stranamente costante
            stdDev[i] = variance > 0 ? Math.sqrt(variance) : 1e-8;
        }
        isFitComplete = true;
        System.out.println("Statistiche globali calcolate su " + count + " campioni.");
    }

    /**
     * FASE 3: STANDARDIZZAZIONE (PC e Smartphone)
     * Applica la formula Z-Score: (x - media) / deviazione_standard
     */
    public double[] standardize(double[] continuousVector) {
        if (!isFitComplete) {
            throw new IllegalStateException("Devi chiamare finalizeStats() o caricare un file prima di standardizzare.");
        }

        double[] standardized = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            standardized[i] = (continuousVector[i] - mean[i]) / stdDev[i];
        }
        return standardized;
    }

    /**
     * EDGE READY: Esporta media e deviazione standard su un file compatto (~80 KB)
     */
    public void exportToFile(String filepath) throws IOException {
        if (!isFitComplete) {
            throw new IllegalStateException("Niente da esportare. Chiudi le statistiche prima.");
        }

        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filepath))) {
            dos.writeInt(dimensions);
            for (int i = 0; i < dimensions; i++) {
                dos.writeDouble(mean[i]);
            }
            for (int i = 0; i < dimensions; i++) {
                dos.writeDouble(stdDev[i]);
            }
        }
        System.out.println("Statistiche VSA salvate in: " + filepath);
    }

    /**
     * EDGE READY: Importa il file pre-calcolato (Da usare su Android/Inferenza)
     */
    public static DatasetStatTracker importFromFile(String filepath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filepath))) {
            int dims = dis.readInt();
            double[] loadedMean = new double[dims];
            double[] loadedStd = new double[dims];

            for (int i = 0; i < dims; i++) {
                loadedMean[i] = dis.readDouble();
            }
            for (int i = 0; i < dims; i++) {
                loadedStd[i] = dis.readDouble();
            }

            System.out.println("Statistiche VSA caricate da: " + filepath);
            return new DatasetStatTracker(dims, loadedMean, loadedStd);
        }
    }
}