package org.antonio;

import org.antonio.bridge.DatasetStatTracker;
import org.antonio.bridge.RandomProjectionBridge;
import org.antonio.domain.PlantVillageTrainer;
import org.antonio.perception.FeatureExtractor;
import org.antonio.perception.PcFeatureExtractor;
import org.antonio.vsa.VsaVector;

public class Trainer {
    public static void main(String[] args) {
        String modelPath = "src/main/resources/mobilenet_savedmodel";
        // Sostituisci questo percorso con la cartella reale di PlantVillage sul tuo PC
//        String plantVillagePath = "../dataset/train";
        String plantVillagePath = "C:\\Users\\Ion\\IdeaProjects\\Vaimee\\VSA\\plant-angine-main\\dataset\\train";
        // C:\Users\Ion\IdeaProjects\Vaimee\VSA\plant-angine-main\plant-vsa-engine

        String outputFolder = "edge_brain"; // Qui verranno salvati i file .bin

        try {
            PcFeatureExtractor realPcEye = new PcFeatureExtractor(modelPath);
        RandomProjectionBridge bridge = new RandomProjectionBridge(FeatureExtractor.CNN_FEATURE_SIZE, 42L);
        DatasetStatTracker tracker = new DatasetStatTracker(VsaVector.DIMENSIONS);

            PlantVillageTrainer autoTrainer = new PlantVillageTrainer(realPcEye, bridge, tracker);

            // Fai partire il macinino!
            autoTrainer.trainDataset(plantVillagePath, outputFolder);

            realPcEye.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
