namespace CLIP;

using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

public class TextProcessor
{
    // Simple word-based tokenization for the sake of demonstration
    public static Tensor<long> TokenizeText(string caption)
    {
        // Split the caption by spaces to simulate tokenization (you should replace this with a real tokenizer)
        var tokens = caption.Split(' ')
                             .Select(word => (long)word.GetHashCode())  // Simulated tokenization (replace with actual token IDs)
                             .ToArray();

        // Create a tensor with token IDs
        return new DenseTensor<long>(tokens, new[] { 1, tokens.Length });
    }
}
