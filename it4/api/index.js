require('dotenv').config();
const express = require("express");
const mongoose = require("mongoose");
const userRouter = require("./Routes/user");
const cors = require('cors');
const path = require('path');
const { userMiddleware } = require('./middleware/user'); // Import the middleware

const app = express();

// Cors configuration
const corsOptions = {
    origin: '*', // Allow all origins (customize in production)
    credentials: true, // Allow credentials (cookies, authorization headers, etc.)
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
};

// Middleware
app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// JWT Secret validation
if (!process.env.JWT_SECRET) {
    console.error('JWT_SECRET is not defined in environment variables');
    process.exit(1);
} 

// Serve static files in production
if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, 'public')));
}

// Basic route for testing
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Server is running' });
});

// Protected routes example
app.get('/api/protected', userMiddleware, (req, res) => {
    res.json({ 
        success: true, 
        message: 'Protected route accessed successfully',
        user: req.user 
    });
});

// API routes
app.use("/api/user", userRouter);

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        message: 'Route not found'
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(err.status || 500).json({
        success: false,
        message: err.message || 'Internal server error'
    });
});

// MongoDB connection options
const mongoOptions = {
    useNewUrlParser: true,
    useUnifiedTopology: true,
};

// MongoDB connection string
const mongoURI = process.env.MONGODB_URI || "mongodb://localhost:27017/Bprep";

async function main() {
    try {
        await mongoose.connect(mongoURI, mongoOptions);
        console.log("MongoDB connected successfully");

        const PORT = process.env.PORT || 3003;
        app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`);
            console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
        });
    } catch (error) {
        console.error("Failed to connect to MongoDB:", error);
        process.exit(1);
    }
}

// Start the server
main();

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
    console.error('Unhandled Promise Rejection:', err);
    process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
    process.exit(1);
});