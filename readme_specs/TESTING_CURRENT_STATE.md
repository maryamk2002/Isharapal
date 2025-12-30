# 🧪 TESTING GUIDE - CURRENT SYSTEM STATE

**Date:** Dec 23, 2025  
**System Version:** V2 (Post-Revert, Pre-Day 2)  
**Active Features:**
- ✅ Hash-based stuck detection (not movement-based)
- ✅ Per-sign confidence thresholds
- ✅ Keypoint filtering enabled
- ✅ Stability voting (5 votes, 3 required = 60%)
- ❌ Motion gating (removed)
- ❌ State machine (removed)

---

## 📋 TEST PLAN

### **TEST 1: Basic Sign Recognition (5 signs)**
**Purpose:** Verify core functionality works

**Signs to Test:**
1. Alif (easy, high accuracy)
2. Bay (easy, high accuracy)
3. Jeem (medium difficulty)
4. Aray (medium difficulty)
5. Gaaf (harder, less training data)

**For Each Sign:**
1. Show sign clearly
2. Hold for 3-4 seconds
3. Remove hand
4. Wait 2 seconds
5. Repeat same sign

**What to Record:**

| Sign | Try 1 Time | Try 1 Correct? | Try 2 Time | Try 2 Correct? | Issues |
|------|------------|----------------|------------|----------------|--------|
| Alif | ___s       | ☐ Yes ☐ No    | ___s       | ☐ Yes ☐ No    |        |
| Bay  | ___s       | ☐ Yes ☐ No    | ___s       | ☐ Yes ☐ No    |        |
| Jeem | ___s       | ☐ Yes ☐ No    | ___s       | ☐ Yes ☐ No    |        |
| Aray | ___s       | ☐ Yes ☐ No    | ___s       | ☐ Yes ☐ No    |        |
| Gaaf | ___s       | ☐ Yes ☐ No    | ___s       | ☐ Yes ☐ No    |        |

**Questions:**
- ☐ Did predictions appear? (Yes/No)
- ☐ Was timing reasonable (<2 seconds)? (Yes/No)
- ☐ Did same sign re-trigger after removing hand? (Yes/No)
- ☐ Any "stuck detected" warnings in logs? (Yes/No)

---

### **TEST 2: Problem Detection (Edge Cases)**

#### **2A: System Getting Stuck**
**Test:** Show sign "Alif", hold for 10 seconds
- ☐ Prediction appears and stays
- ☐ Prediction appears then disappears
- ☐ No prediction at all
- ☐ "Stuck detected" warning in logs

**If stuck, check backend logs for:**
```
[RESET] Buffer cleared...
Stuck sequence detected...
```

---

#### **2B: Flickering Predictions**
**Test:** Show sign "Bay", move hand slightly while holding
- ☐ Prediction stable (same label)
- ☐ Prediction flickers between 2-3 labels
- ☐ Prediction disappears and reappears

**Count:** How many times did the label change? ___

---

#### **2C: Wrong Predictions**
**Test:** Show each sign once, record what system predicted

| Actual Sign | System Predicted | Confidence | Correct? |
|-------------|------------------|------------|----------|
| Alif        |                  | ____%      | ☐ Yes ☐ No |
| Jeem        |                  | ____%      | ☐ Yes ☐ No |
| Bay         |                  | ____%      | ☐ Yes ☐ No |
| Aray        |                  | ____%      | ☐ Yes ☐ No |

**Accuracy:** ___ / ___ correct

---

#### **2D: Slow Response**
**Test:** Show sign "Gaaf", time how long until prediction appears
- Attempt 1: ___s
- Attempt 2: ___s
- Attempt 3: ___s
- **Average:** ___s

**Acceptable?**
- ☐ Yes (< 2 seconds)
- ☐ No (≥ 2 seconds)

---

#### **2E: Not Resetting Between Signs**
**Test:** 
1. Show "Alif" → wait for prediction
2. Remove hand, wait 2s
3. Show "Bay" → check prediction

**Result:**
- ☐ Correctly showed "Bay"
- ☐ Still showing "Alif" (stuck!)
- ☐ Showed wrong sign
- ☐ No prediction

**Check logs for:**
```
[RESET] Buffer cleared: no hands detected
```

---

#### **2F: False Positives (Shoulder Detection)**
**Test:** Don't show any sign, just move shoulders/body
- ☐ No detection (good!)
- ☐ Hand skeleton appears (bad - false positive)
- ☐ Predictions appear (very bad!)

---

### **TEST 3: Performance Monitoring**

**Open Browser Console (F12), look for:**
- FPS: ___ (target: ≥ 12 FPS)
- Prediction time: ___ms (target: < 50ms)

**Backend logs, check for:**
- Inference time: ___ms (target: < 30ms)

---

## 📊 RESULTS SUMMARY

### **Issues Found:**

#### **CRITICAL (System Unusable):**
- ☐ Predictions not appearing at all
- ☐ System stuck after first sign
- ☐ Very slow (>3s for predictions)
- ☐ Accuracy < 50%

#### **MAJOR (Significant Problems):**
- ☐ Flickering predictions (changes >3 times)
- ☐ Wrong predictions frequently (accuracy 50-80%)
- ☐ Slow response (2-3s)
- ☐ Not resetting between signs
- ☐ False positives (detects non-hand objects)

#### **MINOR (Annoying but Workable):**
- ☐ Occasional wrong prediction
- ☐ Slightly slow (1.5-2s)
- ☐ Needs manual reset sometimes
- ☐ Jittery skeleton visualization

#### **NO ISSUES:**
- ☐ Everything works perfectly!

---

## 🎯 SPECIFIC PROBLEMS TO REPORT

**For each issue, describe:**

### **Issue 1:**
- **Type:** (Stuck / Flickering / Wrong / Slow / Other)
- **When:** (Which test? Which sign?)
- **Frequency:** (Every time / Sometimes / Rare)
- **Backend logs:** (Copy relevant error/warning)

### **Issue 2:**
- **Type:** 
- **When:** 
- **Frequency:** 
- **Backend logs:** 

### **Issue 3:**
- **Type:** 
- **When:** 
- **Frequency:** 
- **Backend logs:** 

---

## 📝 OVERALL ASSESSMENT

**Rate the system (1-10):**
- Speed: ___ / 10
- Accuracy: ___ / 10
- Stability: ___ / 10
- Usability: ___ / 10
- **Overall: ___ / 10**

**Is the system usable for a demo?**
- ☐ Yes, good enough as-is
- ☐ Maybe, with minor fixes
- ☐ No, needs significant work

---

## 🚀 NEXT STEPS (Based on Results)

### **If System Works Well (7-10/10):**
→ Consider it done, or add minor polish

### **If System Has Speed Issues (4-6/10):**
→ Proceed with **Day 2: Speed Optimization**
- Optimize interpolation
- Reduce frame size
- Smart buffer management

### **If System Has Stability Issues (4-6/10):**
→ Proceed with **Day 3: UX Enhancement**
- Two-tier stability (tentative/confirmed)
- Adaptive filtering

### **If System Is Broken (<4/10):**
→ Debug specific issues before continuing plan

---

**START TESTING NOW!** 🧪

1. Open: http://localhost:5000/index_v2.html
2. Click "Start Recognition"
3. Follow tests above
4. Report back with results!

