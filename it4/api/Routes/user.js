const { GoogleGenerativeAI } = require("@google/generative-ai");
const { Router } = require("express");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs"); // Add this for password hashing
const { userModel, profileModel, interviewModel } = require("../db");
const { userMiddleware } = require("../middleware/user");
const cors = require("cors");

const JWT_SECRET = process.env.JWT_SECRET || "lmao"; // Should use environment variable in production

const userRouter = Router();

userRouter.use(
  cors({
    origin: "http://127.0.0.1:3000",
    methods: ["GET", "POST", "PUT", "DELETE"],
    credentials: true,
  })
);

const genAI = new GoogleGenerativeAI("");
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

async function generateFollowUpQuestions(question, answer) {
  const prompt = `Based on this question: "${question}" and answer: "${answer}", generate exactly two follow-up questions. 
    Format: Return only the questions, one per line, without numbers or prefixes.`;

  try {
    const result = await model.generateContent(prompt);
    // Clean the response and extract only the questions
    const questions = result.response
      .text()
      .split("\n")
      .map((q) => q.trim())
      // Remove any numbering (1., 2., -, *, etc.)
      .map((q) => q.replace(/^[0-9-.*]\s*\.?\s*/, ""))
      // Remove phrases like "Follow-up questions:", "Here are", etc.
      .filter(
        (q) =>
          q &&
          !q.toLowerCase().includes("follow") &&
          !q.toLowerCase().includes("here")
      )
      .filter((q) => q.trim().endsWith("?"))
      .slice(0, 2);

    return questions;
  } catch (error) {
    console.error("Error generating follow-up questions:", error);
    throw new Error("Failed to generate follow-up questions");
  }
}

const validateSignupInput = (req, res, next) => {
  const { email, password, firstName, lastName } = req.body;

  if (!email || !password || !firstName || !lastName) {
    return res.status(400).json({
      success: false,
      message: "All fields are required",
    });
  }

  if (password.length < 6) {
    return res.status(400).json({
      success: false,
      message: "Password must be at least 6 characters long",
    });
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({
      success: false,
      message: "Invalid email format",
    });
  }

  next();
};

// Signup route
userRouter.post("/signup", validateSignupInput, async (req, res) => {
  try {
    const { email, password, firstName, lastName } = req.body;

    // Check if user already exists
    const existingUser = await userModel.findOne({ email });
    if (existingUser) {
      return res.status(409).json({
        success: false,
        message: "Email already registered",
      });
    }

    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Create user
    const user = await userModel.create({
      email,
      password: hashedPassword,
      firstName,
      lastName,
    });

    // Generate token
    const token = jwt.sign({ id: user._id }, JWT_SECRET, { expiresIn: "24h" });

    res.status(201).json({
      success: true,
      message: "Signup successful",
      token,
    });
  } catch (error) {
    console.error("Signup error:", error);
    res.status(500).json({
      success: false,
      message: "Error during signup",
      error: error.message,
    });
  }
});

// Signin route
userRouter.post("/signin", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: "Email and password are required",
      });
    }

    // Find user
    const user = await userModel.findOne({ email });
    if (!user) {
      return res.status(401).json({
        success: false,
        message: "Invalid credentials",
      });
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({
        success: false,
        message: "Invalid credentials",
      });
    }

    // Generate token
    const token = jwt.sign({ id: user._id }, JWT_SECRET, { expiresIn: "24h" });

    res.json({
      success: true,
      token,
    });
  } catch (error) {
    console.error("Signin error:", error);
    res.status(500).json({
      success: false,
      message: "Error during signin",
      error: error.message,
    });
  }
});

// Create/Update Profile route

userRouter.get("/profile", userMiddleware, async (req, res) => {
  try {
    const studentId = req.user.id;

    if (!studentId) {
      return res.status(400).json({
        success: false,
        message: "Student ID is required",
      });
    }

    const profile = await profileModel.findOne({ studentId });

    if (!profile) {
      return res.status(404).json({
        success: false,
        message: "Profile not found",
      });
    }

    res.json({
      success: true,
      profile,
    });
  } catch (error) {
    console.error("Error retrieving profile:", error);
    res.status(500).json({
      success: false,
      message: "Error retrieving profile",
      error: error.message,
    });
  }
});

userRouter.post("/profile", async (req, res) => {
  try {
    const { studentId, fullName, phoneNumber, CATscore, gradSchool } = req.body;

    if (!fullName) {
      return res.status(400).json({
        success: false,
        message: "Full name is required",
      });
    }

    if (!studentId) {
      return res.status(400).json({
        success: false,
        message: "Student ID is required",
      });
    }

    const profile = await profileModel.findOneAndUpdate(
      { studentId },
      {
        studentId,
        fullName,
        phoneNumber,
        CATscore,
        gradSchool,
        updatedAt: Date.now(),
      },
      { upsert: true, new: true }
    );

    res.json({
      success: true,
      message: "Profile updated successfully",
      profile,
    });
  } catch (error) {
    console.error("Profile update error:", error);
    res.status(500).json({
      success: false,
      message: "Error updating profile",
      error: error.message,
    });
  }
});

// Submit Interview route
userRouter.post("/interview", userMiddleware, async (req, res) => {
  try {
    const { motivation, habits } = req.body;

    if (!motivation || !habits) {
      return res.status(400).json({
        success: false,
        message: "All interview fields are required",
      });
    }

    // Generate follow-up questions with clean formatting
    const motivationFollowUps = await generateFollowUpQuestions(
      "What motivates you?",
      motivation
    );
    const habitsFollowUps = await generateFollowUpQuestions(
      "What are your habits?",
      habits
    );

    // Create or update interview
    const interview = await interviewModel.findOneAndUpdate(
      { studentId: req.user.id },
      {
        studentId: req.user.id,
        motivation,
        habits,
        followUpQuestions: [
          ...motivationFollowUps.map((q) => ({ question: q, answer: "" })),
          ...habitsFollowUps.map((q) => ({ question: q, answer: "" })),
        ],
        updatedAt: Date.now(),
      },
      { upsert: true, new: true }
    );

    res.json({
      success: true,
      message: "Interview responses submitted successfully",
      interview,
      followUpQuestions: [...motivationFollowUps, ...habitsFollowUps],
    });
  } catch (error) {
    console.error("Interview submission error:", error);
    res.status(500).json({
      success: false,
      message: "Error submitting interview responses",
      error: error.message,
    });
  }
});

userRouter.post("/interview-followup", userMiddleware, async (req, res) => {
  try {
    const { answers } = req.body;

    if (!answers || !Array.isArray(answers)) {
      return res.status(400).json({
        success: false,
        message: "Answers array is required",
      });
    }

    const existingInterview = await interviewModel.findOne({
      studentId: req.user.id,
    });

    if (!existingInterview) {
      return res.status(404).json({
        success: false,
        message:
          "No interview found. Please submit initial interview responses first.",
      });
    }

    // Update the answers while keeping the original questions
    const updatedQuestions = existingInterview.followUpQuestions.map(
      (qaPair, index) => ({
        question: qaPair.question,
        answer: answers[index] || "",
      })
    );

    const updatedInterview = await interviewModel.findOneAndUpdate(
      { studentId: req.user.id },
      {
        $set: {
          followUpQuestions: updatedQuestions,
          updatedAt: new Date(),
        },
      },
      { new: true }
    );

    res.json({
      success: true,
      message: "Follow-up answers submitted successfully",
      interview: updatedInterview,
    });
  } catch (error) {
    console.error("Follow-up submission error:", error);
    res.status(500).json({
      success: false,
      message: "Error submitting follow-up answers",
      error: error.message,
    });
  }
});

userRouter.get("/interview", userMiddleware, async (req, res) => {
  try {
    const interview = await interviewModel.findOne({
      studentId: req.user.id,
    });

    if (!interview) {
      return res.status(404).json({
        success: false,
        message: "No interview found for this user",
      });
    }

    res.json({
      success: true,
      interview,
    });
  } catch (error) {
    console.error("Interview fetch error:", error);
    res.status(500).json({
      success: false,
      message: "Error fetching interview data",
      error: error.message,
    });
  }
});

// Download interviews route (admin only)
userRouter.get("/interviews/download", userMiddleware, async (req, res) => {
  try {
    // Optional: Check if user is admin
    if (!req.user.isAdmin) {
      return res.status(403).json({
        success: false,
        message: "Access denied. Admin privileges required.",
      });
    }

    // Fetch all interviews
    const interviews = await interviewModel
      .find({})
      .populate("studentId", "name email");

    // Format data for CSV
    const csvData = interviews.map((interview) => ({
      studentName: interview.studentId.name,
      studentEmail: interview.studentId.email,
      motivation: interview.motivation,
      habits: interview.habits,
      followUpQuestions: interview.followUpQuestions
        .map((q) => `Q: ${q.question}\nA: ${q.answer}`)
        .join(" | "),
      createdAt: interview.createdAt,
      updatedAt: interview.updatedAt,
    }));

    // Convert to CSV string
    const fields = [
      "studentName",
      "studentEmail",
      "motivation",
      "habits",
      "followUpQuestions",
      "createdAt",
      "updatedAt",
    ];
    const csv = [
      fields.join(","),
      ...csvData.map((row) =>
        fields
          .map((field) => `"${String(row[field]).replace(/"/g, '""')}"`)
          .join(",")
      ),
    ].join("\n");

    // Set headers for file download
    res.setHeader("Content-Type", "text/csv");
    res.setHeader("Content-Disposition", "attachment; filename=interviews.csv");

    // Send CSV
    res.send(csv);
  } catch (error) {
    console.error("Interview download error:", error);
    res.status(500).json({
      success: false,
      message: "Error downloading interviews",
      error: error.message,
    });
  }
});

module.exports = userRouter;
