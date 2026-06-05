package org.antonio;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class MatrixExporter {

    public static void main(String[] args) {
        int cnnFeatureSize = 1280;
        int vsaDimensions = 10000;
        long seed = 42L; // IL TUO SEED ESATTO

        String filename = "vsa_projection_matrix.csv";

        System.out.println("Generazione della matrice in corso...");
        Random random = new Random(seed);

        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            for (int i = 0; i < cnnFeatureSize; i++) {
                StringBuilder row = new StringBuilder();
                for (int d = 0; d < vsaDimensions; d++) {
                    // Generiamo il booleano esattamente come nel tuo RandomProjectionBridge
                    boolean bit = random.nextBoolean();

                    // Convertiamo true in "1" e false in "-1" per Python
                    row.append(bit ? "1" : "-1");

                    if (d < vsaDimensions - 1) {
                        row.append(",");
                    }
                }
                writer.println(row.toString());
            }
            System.out.println("✅ Matrice esportata con successo in: " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}