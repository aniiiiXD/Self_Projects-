
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const fileModel = require('./DB/File'); 

const app = express();

const corsOptions = {
  origin: 'http://localhost:5173', 
  methods: 'GET,POST', 
  allowedHeaders: 'Content-Type, Authorization', 
};
app.use(cors(corsOptions));
app.use(express.json());


mongoose.connect('mongodb://localhost:27017/Vectorr', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('MongoDB connection error:', err));


const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  },
});

const upload = multer({ storage: storage });


app.post('/upload-files', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    const fileDoc = new fileModel({
      title: req.file.originalname,
      path: req.file.path,
    });

    await fileDoc.save();
    res.status(200).json({ message: 'File uploaded successfully', file: fileDoc });
  } catch (error) {
    console.error('Error uploading file:', error);
    res.status(500).json({ message: 'Error uploading file' });
  }
});

app.listen(3002, () => {
  console.log("Server is running on port 3002");
});
