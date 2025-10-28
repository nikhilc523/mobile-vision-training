# Dataset Validation and Cleanup Report

**Generated:** 2025-10-28 00:59:06

## Summary

### URFD Dataset

- **Fall videos:** 31
- **ADL videos:** 32
- **Total:** 63

### Le2i Dataset

- **Total videos:** 190
- **Total annotations:** 130
- **Videos without annotations:** 60
- **Annotations without videos:** 0

### Le2i Scene Breakdown

| Scene | Videos | Annotations | Issues |
|-------|--------|-------------|--------|
| Coffee_room_01 | 48 | 48 | ✅ None |
| Coffee_room_02 | 22 | 22 | ✅ None |
| Home_01 | 30 | 30 | ✅ None |
| Home_02 | 30 | 30 | ✅ None |
| Lecture room | 27 | 0 | 27 w/o ann |
| Office | 33 | 0 | 33 w/o ann |

## Cleanup Operations

**Mode:** Dry run (no changes made)

- **Files identified for cleanup:** 68
- **Empty folders found:** 0
- **Files deleted:** 0
- **Folders removed:** 0

### Files Identified for Cleanup

**.DS_Store** (4 files):

- 🔍 Found: `.DS_Store`
- 🔍 Found: `urfd/.DS_Store`
- 🔍 Found: `le2i/.DS_Store`
- 🔍 Found: `le2i/Home_01/.DS_Store`

**.zip** (64 files):

- 🔍 Found: `urfd/falls/fall-18-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-29-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-19-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-28-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-30-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-22-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-25-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-13-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-14-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-01-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-06-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-24-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-23-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-07-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-30-cam1-rgb.zip`
- 🔍 Found: `urfd/falls/fall-15-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-12-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-21-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-26-cam0-rgb.zip`
- 🔍 Found: `urfd/falls/fall-02-cam0-rgb.zip`
- ... and 44 more

## Issues Detected

### Videos Without Annotations (60)

- ⚠️  `Lecture room/video (13).avi`
- ⚠️  `Lecture room/video (10).avi`
- ⚠️  `Lecture room/video (17).avi`
- ⚠️  `Lecture room/video (27).avi`
- ⚠️  `Lecture room/video (26).avi`
- ⚠️  `Lecture room/video (20).avi`
- ⚠️  `Lecture room/video (6).avi`
- ⚠️  `Lecture room/video (19).avi`
- ⚠️  `Lecture room/video (3).avi`
- ⚠️  `Lecture room/video (24).avi`
- ⚠️  `Lecture room/video (5).avi`
- ⚠️  `Lecture room/video (8).avi`
- ⚠️  `Lecture room/video (18).avi`
- ⚠️  `Lecture room/video (21).avi`
- ⚠️  `Lecture room/video (16).avi`
- ⚠️  `Lecture room/video (9).avi`
- ⚠️  `Lecture room/video (14).avi`
- ⚠️  `Lecture room/video (11).avi`
- ⚠️  `Lecture room/video (25).avi`
- ⚠️  `Lecture room/video (1).avi`
- ... and 40 more
