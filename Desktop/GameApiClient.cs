using System.Text;
using System.Text.Json;

namespace TetrisStudio;

internal static class GameApiClient
{
    public static async Task<JsonDocument> Post(HttpClient http, Uri url, object data)
    {
        // Buffer the small command so HttpClient sends Content-Length. Python's
        // BaseHTTPRequestHandler does not decode chunked request bodies for us.
        using var content = new StringContent(JsonSerializer.Serialize(data), Encoding.UTF8, "application/json");
        using var response = await http.PostAsync(url, content);
        string body = await response.Content.ReadAsStringAsync();
        var document = JsonDocument.Parse(body);
        if (response.IsSuccessStatusCode) return document;
        string error = document.RootElement.TryGetProperty("error", out var value)
            ? value.GetString() ?? "Game request failed" : response.ReasonPhrase ?? "Game request failed";
        document.Dispose();
        throw new InvalidOperationException(error);
    }
}
