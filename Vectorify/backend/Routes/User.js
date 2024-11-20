const { Router } = require("express");
const { fileModel } = require("../DB/db");
const multer = require("multer");

const userRouter = Router();





userRouter.post("/api/upload-files", upload.single('file'), async (req, res) => {
  
});

module.exports = userRouter;
