const jwt = require('jsonwebtoken');

// Best practice: Use environment variable for JWT secret
const JWT_SECRET = process.env.JWT_SECRET || 's3cret!prep'; // Remember to use a strong secret in production!

const userMiddleware = (req, res, next) => {
    try {
        
        const authHeader = req.headers.authorization;
        if (!authHeader) {
            return res.status(401).json({
                success: false,
                message: 'No authorization header found'
            });
        }

        // Extract the token (Bearer token format)
        const token = authHeader.startsWith('Bearer ') 
            ? authHeader.slice(7) 
            : authHeader;

        if (!token) {
            return res.status(401).json({
                success: false,
                message: 'No token provided'
            });
        }

        // Verify the token
        const decoded = jwt.verify(token, JWT_SECRET);
        
        // Add user data to request object
        req.user = {
            id: decoded.id,
            // Add any other necessary user data from token
        };

        next();
    } catch (error) {
        // Handle different types of JWT errors
        if (error.name === 'JsonWebTokenError') {
            return res.status(401).json({
                success: false,
                message: 'Invalid token'
            });
        }
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({
                success: false,
                message: 'Token has expired'
            });
        }

        // Handle any other errors
        return res.status(500).json({
            success: false,
            message: 'Authentication error',
            error: error.message
        });
    }
};

module.exports = {
    userMiddleware
};