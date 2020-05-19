import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class IrisPrediction {
    public static void main(String[] args) throws IOException {
        //loading the model
        MultiLayerNetwork model= ModelSerializer.restoreMultiLayerNetwork(new File("irisModelPrediction.zip"));
        System.out.println("Prediction");
        INDArray inputTopredict= Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2},
                {4.9,3.0,1.4,0.2},
                {6.7,3.1,4.4,1.4},
                {5.6,3.0,4.5,1.5},
                {6.0,3.0,4.8,1.8},
                {6.9,3.1,5.4,2.1}
        }); //matrice de double
        INDArray output=model.output(inputTopredict);
        System.out.println("result : "+output);

        int[] classes=output.argMax(1).toIntVector();
        String[] classesNames={"Iris-setosa","Iris-versicolor","Iris-verginica"};


        for(int i=0;i<classes.length;i++){
            System.out.println("classe : "+classesNames[classes[i]]);
        }



    }
}
