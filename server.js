const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const sharp = require('sharp');
const { PythonShell } = require('python-shell');
const helmet = require('helmet');
const { RateLimiterMemory } = require('rate-limiter-flexible');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

// Rate limiting
const rateLimiter = new RateLimiterMemory({
  keyPrefix: 'deeptrust_api',
  points: 10, // Number of requests
  duration: 60, // Per 60 seconds
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: ['chrome-extension://*', 'https://*'],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json({ limit: '10mb' }));

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/') || file.mimetype.startsWith('video/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image and video files are allowed'));
    }
  }
});

// Create uploads directory if it doesn't exist
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

// Helper function to run PyDeepFakeDet
async function runDeepFakeDetection(imagePath) {
  return new Promise((resolve, reject) => {
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      scriptPath: path.join(__dirname, 'python'),
      args: [imagePath]
    };

    PythonShell.run('detect_deepfake.py', options, (err, results) => {
      if (err) {
        console.error('Python script error:', err);
        reject(err);
      } else {
        resolve(results[0]);
      }
    });
  });
}

// Helper function to generate explanation using LLM
async function generateExplanation(detectionResult, imageFeatures) {
  const prompt = `
    Based on the deepfake detection analysis, explain why this image was flagged as potentially fake.
    
    Detection confidence: ${detectionResult.confidence}%
    Technical indicators: ${JSON.stringify(detectionResult.indicators || {})}
    
    Provide 2-4 specific, easy-to-understand reasons in this format:
    ["Facial edges appear blurred and inconsistent", "Lighting on face doesn't match background", "Mouth movement seems unnatural"]
    
    Focus on visual artifacts that a person could potentially notice.
  `;

  try {
    // If OpenAI API key is available, use GPT
    if (process.env.OPENAI_API_KEY) {
      const OpenAI = require('openai');
      const openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY
      });

      const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: "You are an expert at explaining deepfake detection results in simple terms."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        max_tokens: 150,
        temperature: 0.3
      });

      const explanation = response.choices[0].message.content;
      try {
        return JSON.parse(explanation);
      } catch {
        return explanation.split('\n').filter(line => line.trim()).slice(0, 4);
      }
    } else {
      // Fallback to rule-based explanations
      const reasons = [];
      const confidence = detectionResult.confidence;

      if (confidence > 80) {
        reasons.push("Strong facial manipulation artifacts detected");
        reasons.push("Inconsistent lighting patterns on face");
      }
      if (confidence > 60) {
        reasons.push("Blurred or unnatural facial edges");
        reasons.push("Temporal inconsistencies in video frames");
      }
      if (confidence > 40) {
        reasons.push("Subtle facial feature anomalies");
        reasons.push("Possible digital compression artifacts");
      }

      return reasons.slice(0, Math.min(4, Math.max(2, Math.floor(confidence / 20))));
    }
  } catch (error) {
    console.error('Error generating explanation:', error);
    return ["AI model detected suspicious patterns", "Further analysis recommended"];
  }
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'DeepTrust API',
    model: 'PyDeepFakeDet',
    timestamp: new Date().toISOString()
  });
});

// Main deepfake detection endpoint for URL-based images
app.post('/api/detect-deepfake', async (req, res) => {
  try {
    // Rate limiting
    await rateLimiter.consume(req.ip);

    const { imageUrl, source } = req.body;

    if (!imageUrl) {
      return res.status(400).json({
        success: false,
        error: 'Image URL is required'
      });
    }

    console.log(`Analyzing image from ${source}: ${imageUrl}`);

    // Download the image
    const response = await axios({
      method: 'GET',
      url: imageUrl,
      responseType: 'arraybuffer',
      timeout: 10000,
      headers: {
        'User-Agent': 'DeepTrust/1.0 (+https://deeptrust.ai)'
      }
    });

    // Process image with Sharp
    const imageBuffer = Buffer.from(response.data);
    const processedImage = await sharp(imageBuffer)
      .resize(256, 256, { fit: 'cover' })
      .jpeg({ quality: 90 })
      .toBuffer();

    // Save temporary file
    const tempPath = path.join(__dirname, 'uploads', `temp_${Date.now()}.jpg`);
    fs.writeFileSync(tempPath, processedImage);

    try {
      // Run PyDeepFakeDet
      const detectionResult = await runDeepFakeDetection(tempPath);
      
      // Generate explanation
      const explanationReasons = await generateExplanation(detectionResult, {
        source,
        size: `${response.headers['content-length'] || 'unknown'} bytes`
      });

      // Clean up temp file
      fs.unlinkSync(tempPath);

      res.json({
        success: true,
        confidence: Math.round(detectionResult.confidence || 0),
        isDeepfake: detectionResult.confidence > 50,
        reasons: explanationReasons,
        model: 'PyDeepFakeDet',
        timestamp: new Date().toISOString()
      });

    } catch (detectionError) {
      console.error('Detection error:', detectionError);
      fs.unlinkSync(tempPath); // Clean up on error
      
      res.json({
        success: true,
        confidence: 0,
        isDeepfake: false,
        reasons: ['Detection service temporarily unavailable'],
        model: 'PyDeepFakeDet',
        timestamp: new Date().toISOString()
      });
    }

  } catch (error) {
    console.error('API Error:', error);
    
    if (error.name === 'RateLimiterRes') {
      return res.status(429).json({
        success: false,
        error: 'Rate limit exceeded. Please try again later.'
      });
    }

    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

// Upload-based detection endpoint for video frames
app.post('/api/detect-deepfake-upload', upload.single('image'), async (req, res) => {
  try {
    await rateLimiter.consume(req.ip);

    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded'
      });
    }

    const { source } = req.body;
    console.log(`Analyzing uploaded file from ${source}`);

    try {
      // Process uploaded file
      const processedImage = await sharp(req.file.path)
        .resize(256, 256, { fit: 'cover' })
        .jpeg({ quality: 90 })
        .toBuffer();

      const processedPath = req.file.path + '_processed.jpg';
      fs.writeFileSync(processedPath, processedImage);

      // Run detection
      const detectionResult = await runDeepFakeDetection(processedPath);
      const explanationReasons = await generateExplanation(detectionResult, {
        source,
        type: 'video_frame'
      });

      // Clean up files
      fs.unlinkSync(req.file.path);
      fs.unlinkSync(processedPath);

      res.json({
        success: true,
        confidence: Math.round(detectionResult.confidence || 0),
        isDeepfake: detectionResult.confidence > 50,
        reasons: explanationReasons,
        model: 'PyDeepFakeDet',
        timestamp: new Date().toISOString()
      });

    } catch (detectionError) {
      console.error('Upload detection error:', detectionError);
      
      // Clean up files on error
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }

      res.json({
        success: true,
        confidence: 0,
        isDeepfake: false,
        reasons: ['Video analysis temporarily unavailable'],
        model: 'PyDeepFakeDet',
        timestamp: new Date().toISOString()
      });
    }

  } catch (error) {
    console.error('Upload API Error:', error);
    res.status(500).json({
      success: false,
      error: 'Upload processing failed'
    });
  }
});

// Start server
app.listen(port, () => {
  console.log(`🛡️  DeepTrust API server running on port ${port}`);
  console.log(`📊 Health check: http://localhost:${port}/api/health`);
  console.log(`🔍 Detection endpoint: http://localhost:${port}/api/detect-deepfake`);
});
