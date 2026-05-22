/**
 * ChestModel - A wrapper for the PyTorch ConvNeXt Large model
 * This is a TypeScript representation of the PyTorch model architecture
 * The actual inference is done via a backend API call
 */

export interface DetectionResults {
  [key: string]: number;
  atelectasis: number;
  cardiomegaly: number;
  effusion: number;
  infiltration: number;
  mass: number;
  nodule: number;
  pneumonia: number;
  pneumothorax: number;
  consolidation: number;
  edema: number;
  emphysema: number;
  fibrosis: number;
  pleural_thickening: number;
  hernia: number;
  no_finding: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BoundingBoxes {
  [key: string]: BoundingBox;
}

export interface ModelMetadata {
  age: number;
  sex: number; // 1 for male, 0 for female
  viewPosition: number; // 0 for PA, 1 for AP
}

export const DISEASE_LABELS: Record<string, string> = {
  atelectasis: 'Atelectasis',
  cardiomegaly: 'Cardiomegaly',
  effusion: 'Effusion',
  infiltration: 'Infiltration',
  mass: 'Mass',
  nodule: 'Nodule',
  pneumonia: 'Pneumonia',
  pneumothorax: 'Pneumothorax',
  consolidation: 'Consolidation',
  edema: 'Edema',
  emphysema: 'Emphysema',
  fibrosis: 'Fibrosis',
  pleural_thickening: 'Pleural Thickening',
  hernia: 'Hernia',
  no_finding: 'No Finding'
};

/**
 * API endpoint for the real model backend
 * This needs to be set to your actual API endpoint when deployed
 */
const MODEL_API_ENDPOINT = process.env.NEXT_PUBLIC_MODEL_API_ENDPOINT || 'http://localhost:5000/predict';

/**
 * ChestModel class for managing interactions with the chest X-ray model
 */
class ChestModel {
  private modelPath: string = "/models/best_model_epoch_3_auroc_9004.pth";
  private useMockData: boolean = false; // Now using the real API
  private isProcessing: boolean = false; // Flag to prevent concurrent requests

  /**
   * Make a prediction using the chest X-ray model
   * @param imageUrl URL of the X-ray image
   * @param metadata Patient metadata (age, sex, view position)
   * @returns Detection results and bounding boxes
   */
  async predict(imageUrl: string, metadata: ModelMetadata): Promise<{ detections: DetectionResults; boxes: BoundingBoxes }> {
    // Prevent concurrent calls to avoid infinite loops
    if (this.isProcessing) {
      console.log("Another prediction is already in progress, skipping this request");
      throw new Error("Another prediction is already in progress");
    }
    
    this.isProcessing = true;
    
    try {
      console.log("Using real model API endpoint:", MODEL_API_ENDPOINT);
      
      // If we're using a data URL, we'll need to convert it to a blob for upload
      let imageBlob: Blob;
      if (imageUrl.startsWith('data:')) {
        imageBlob = await this.dataURLtoBlob(imageUrl);
      } else {
        // Fetch the image from the URL
        const response = await fetch(imageUrl);
        imageBlob = await response.blob();
      }

      // Create form data for API request
      const formData = new FormData();
      formData.append('image', imageBlob, 'image.jpg');
      formData.append('age', metadata.age.toString());
      formData.append('sex', metadata.sex.toString());
      formData.append('view_position', metadata.viewPosition.toString());

      // Make API request with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      try {
        const response = await fetch(MODEL_API_ENDPOINT, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Check for errors
        if (!response.ok) {
          throw new Error(`API error: ${response.status} ${response.statusText}`);
        }

        // Parse response
        const result = await response.json();
        return {
          detections: result.detections,
          boxes: result.boxes
        };
      } catch (fetchError) {
        console.error("Fetch error:", fetchError);
        
        // If the model server is unavailable, fall back to API route
        console.log("Falling back to internal API route");
        const fallbackResponse = await fetch('/api/model', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ metadata }),
        });
        
        if (!fallbackResponse.ok) {
          throw new Error(`Fallback API error: ${fallbackResponse.status}`);
        }
        
        const fallbackResult = await fallbackResponse.json();
        return {
          detections: fallbackResult.detections,
          boxes: fallbackResult.boxes
        };
      }
    } catch (error) {
      console.error("Error in model prediction:", error);
      throw error;
    } finally {
      // Always reset the processing flag
      this.isProcessing = false;
    }
  }

  /**
   * Convert data URL to Blob
   * @param dataURL Data URL
   * @returns Blob
   */
  private async dataURLtoBlob(dataURL: string): Promise<Blob> {
    try {
      const response = await fetch(dataURL);
      const blob = await response.blob();
      return blob;
    } catch (error) {
      console.error("Error converting data URL to blob:", error);
      throw error;
    }
  }
}

export default new ChestModel(); 