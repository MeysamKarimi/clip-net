namespace CLIP;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

public class CLIPModel
{
    private InferenceSession _session;

    public CLIPModel(string modelPath)
    {
        // Load the ONNX model
        _session = new InferenceSession(modelPath);
    }

    public Tensor<float> GetImageEmbedding(Tensor<float> imageTensor)
    {
        // Prepare input for the model
        var inputMeta = _session.InputMetadata;
        var inputName = inputMeta.First().Key; // Assuming the input is named "image"

        // Create a NamedOnnxValue
        var namedInput = NamedOnnxValue.CreateFromTensor(inputName, imageTensor);

        // Run inference
        var results = _session.Run(new[] { namedInput });

        // Get the output tensor
        return results.First().AsTensor<float>();
    }

    public Tensor<float> GetTextEmbedding(Tensor<long> textTensor)
    {
        var inputMeta = _session.InputMetadata;
        var inputName = inputMeta.First().Key;

        // Convert Tensor<long> to Tensor<float>
        var floatTensor = ConvertTensorToFloat(textTensor);

        // Dynamically adjust the reshape size
        if (floatTensor.Length < 150528)
        {
            // Example: Text embeddings with size 18
            return ReshapeToRank4(floatTensor, batchSize: 1, channels: 1, height: 1, width: (int)floatTensor.Length);
        }
        else if (floatTensor.Length == 150528)
        {
            // Example: Image embeddings with size 1x3x224x224
            return ReshapeToRank4(floatTensor, batchSize: 1, channels: 3, height: 224, width: 224);
        }
        else
        {
            throw new ArgumentException($"Unexpected tensor size: {floatTensor.Length}");
        }
    }


    private DenseTensor<float> ReshapeToRank4(DenseTensor<float> inputTensor, int batchSize, int channels, int height, int width)
    {
        Console.WriteLine($"Original Tensor Dimensions: {string.Join(", ", inputTensor.Dimensions.Length)}");
        Console.WriteLine($"Original Tensor Length: {inputTensor.Length}");

        int expectedSize = batchSize * channels * height * width;
        if (inputTensor.Length != expectedSize)
        {
            throw new ArgumentException($"The total size of the tensor ({inputTensor.Length}) does not match the specified dimensions ({expectedSize}).");
        }

        // Create the reshaped tensor
        var reshapedTensor = new DenseTensor<float>(new[] { batchSize, channels, height, width });
        var flatInput = inputTensor.ToArray();

        for (int i = 0; i < flatInput.Length; i++)
        {
            reshapedTensor.Buffer.Span[i] = flatInput[i];
        }

        return reshapedTensor;
    }

    private DenseTensor<float> ConvertTensorToFloat(Tensor<long> longTensor)
    {
        if (longTensor is not DenseTensor<long> denseLongTensor)
        {
            throw new ArgumentException("The input tensor must be a DenseTensor<long>.");
        }

        // Create a float tensor with the same shape
        var floatTensor = new DenseTensor<float>(denseLongTensor.Dimensions);

        // Convert values
        for (int i = 0; i < denseLongTensor.Length; i++)
        {
            floatTensor.Buffer.Span[i] = (float)denseLongTensor.Buffer.Span[i];
        }

        return floatTensor;
    }
}
