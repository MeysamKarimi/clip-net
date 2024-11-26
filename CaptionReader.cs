namespace CLIP;

public class CaptionReader
{
    public static Dictionary<string, List<string>> LoadCaptions(string captionFilePath)
    {
        var captions = new Dictionary<string, List<string>>();

        try
        {
            // Read all lines from the caption file
            var lines = File.ReadAllLines(captionFilePath);

            foreach (var line in lines)
            {
                // Split by tab to separate image filename and caption
                var parts = line.Split(',');

                if (parts.Length == 2 && parts[0] != "image")
                {
                    var imageName = parts[0].Split('#')[0];  // Extract image name
                    var caption = parts[1];

                    // If this image doesn't have any captions yet, create a new list
                    if (!captions.ContainsKey(imageName))
                    {
                        captions[imageName] = new List<string>();
                    }

                    // Add the caption to the list of captions for this image
                    captions[imageName].Add(caption);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error reading captions: {ex.Message}");
        }

        return captions;
    }
}