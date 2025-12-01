# TensorBoard Guide - How to Get Screenshots for Wiki

## ✅ TensorBoard is Running!

**URL:** http://localhost:6006

TensorBoard is now running and showing your training history!

---

## 📊 What You'll See in TensorBoard

### Available Metrics (7 Categories)

1. **epoch_loss** - Training vs Validation Loss
2. **epoch_accuracy** - Training vs Validation Accuracy  
3. **epoch_auc** - Training vs Validation ROC-AUC
4. **epoch_precision** - Training vs Validation Precision
5. **epoch_recall** - Training vs Validation Recall
6. **epoch_f1** - Training vs Validation F1 Score
7. **learning_rate** - Learning Rate Schedule

---

## 🎯 How to Take Screenshots for Your Professor

### Step 1: Navigate TensorBoard

1. **Open:** http://localhost:6006 (should already be open)
2. **Click "SCALARS" tab** at the top
3. You'll see all your training metrics organized by category

### Step 2: Customize the View

**For Better Screenshots:**

1. **Smooth curves:** Adjust the "Smoothing" slider (try 0.6 for cleaner curves)
2. **Toggle runs:** Show/hide train vs validation
3. **Zoom:** Click and drag to zoom into specific epochs
4. **Full screen:** Click the expand icon on each graph

### Step 3: Take Screenshots

**Recommended Screenshots for Wiki:**

#### Screenshot 1: Loss Curves
1. Find **"epoch_loss"** section
2. Shows both train and validation loss
3. Take screenshot (Cmd+Shift+4 on Mac)
4. Save as: `tensorboard_loss_curves.png`

#### Screenshot 2: Accuracy Curves
1. Find **"epoch_accuracy"** section
2. Shows both train and validation accuracy
3. Take screenshot
4. Save as: `tensorboard_accuracy_curves.png`

#### Screenshot 3: F1 Score Curves
1. Find **"epoch_f1"** section
2. Shows both train and validation F1
3. Take screenshot
4. Save as: `tensorboard_f1_curves.png`

#### Screenshot 4: Learning Rate Schedule
1. Find **"learning_rate"** section
2. Shows how LR changed over epochs
3. Take screenshot
4. Save as: `tensorboard_learning_rate.png`

#### Screenshot 5: All Metrics Overview
1. Scroll to see all graphs at once
2. Take a full-page screenshot
3. Save as: `tensorboard_overview.png`

### Step 4: Save Screenshots

**Save all screenshots to:**
```
docs/wiki_assets/tensorboard_screenshots/
```

Create the folder:
```bash
mkdir -p docs/wiki_assets/tensorboard_screenshots
```

Then move your screenshots there.

---

## 🎨 TensorBoard Tips for Best Screenshots

### 1. Adjust Smoothing
- **Smoothing = 0:** Raw data (noisy)
- **Smoothing = 0.6:** Recommended (smooth but accurate)
- **Smoothing = 0.9:** Very smooth (may hide details)

### 2. Compare Train vs Validation
- Both lines should be visible
- Orange = one metric, Blue = another
- Hover to see exact values

### 3. Zoom to Important Regions
- Click and drag to zoom
- Focus on convergence region (epochs 40-74)
- Double-click to reset zoom

### 4. Use Full Screen Mode
- Click expand icon (⛶) on each graph
- Gives cleaner, larger screenshots
- Better for presentations

### 5. Download Data
- Click "Show data download links" at bottom left
- Can download CSV or JSON
- Useful for custom plots

---

## 📸 Screenshot Checklist

For your professor, take these screenshots:

- [ ] **Loss curves** (train vs validation)
- [ ] **Accuracy curves** (train vs validation)
- [ ] **F1 score curves** (train vs validation)
- [ ] **ROC-AUC curves** (train vs validation)
- [ ] **Learning rate schedule**
- [ ] **Overview** (all metrics visible)

---

## 🚀 How to Stop TensorBoard

When you're done taking screenshots:

**Option 1: In Terminal**
```bash
# Press Ctrl+C in the terminal where TensorBoard is running
```

**Option 2: Kill Process**
```bash
# Find the process
ps aux | grep tensorboard

# Kill it
kill <PID>
```

---

## 📝 Adding Screenshots to Wiki

### Option 1: Upload to GitHub First

1. Save screenshots to `docs/wiki_assets/tensorboard_screenshots/`
2. Commit and push:
   ```bash
   git add docs/wiki_assets/tensorboard_screenshots/
   git commit -m "Add TensorBoard screenshots"
   git push
   ```
3. In wiki, use:
   ```markdown
   ![Loss Curves](https://raw.githubusercontent.com/nikhilc523/mobile-vision-training/main/docs/wiki_assets/tensorboard_screenshots/tensorboard_loss_curves.png)
   ```

### Option 2: Direct Upload to Wiki

1. Go to GitHub Wiki editor
2. Drag and drop screenshots directly
3. GitHub will host them automatically

---

## 🎓 What Your Professor Will See

**Real TensorBoard Screenshots Show:**

✅ **Professional tool** - Industry-standard visualization  
✅ **Interactive data** - Real training logs, not just plots  
✅ **Detailed metrics** - All metrics tracked over time  
✅ **Smooth curves** - Adjustable smoothing for clarity  
✅ **Exact values** - Hover to see precise numbers  
✅ **Reproducible** - Can regenerate from logs anytime  

**This proves:**
- You used proper ML tools (TensorBoard)
- You tracked metrics during training
- You understand model convergence
- You follow industry best practices

---

## 💡 Pro Tips

### Tip 1: Compare Multiple Runs
If you train multiple models, TensorBoard can show them all:
```bash
tensorboard --logdir=logs
```
All runs in `logs/` folder will appear!

### Tip 2: Share TensorBoard Online
Use TensorBoard.dev to share publicly:
```bash
tensorboard dev upload --logdir logs/bilstm_fall_detection_*
```
Gives you a public URL to share!

### Tip 3: Export High-Res Images
For publications, use TensorBoard's download feature:
1. Click graph
2. Click "..." menu
3. Select "Download as SVG" (vector graphics)

---

## 🎉 You're All Set!

**TensorBoard is running at:** http://localhost:6006

**Next steps:**
1. ✅ Explore the metrics in TensorBoard
2. ✅ Take screenshots of key graphs
3. ✅ Save to `docs/wiki_assets/tensorboard_screenshots/`
4. ✅ Add to your wiki page
5. ✅ Impress your professor! 🚀

---

**Questions?**
- TensorBoard docs: https://www.tensorflow.org/tensorboard
- TensorBoard tutorial: https://www.tensorflow.org/tensorboard/get_started

