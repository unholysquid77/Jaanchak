# What we need from the forensic datasets

Status: **not blocking.** The pipeline runs end-to-end on a public cross-domain
benchmark, and Levels 0-2 can be built and validated without any forensic images. This
document exists so the questions get asked early rather than discovered in week four.

---

## The one question that actually matters

> **How many distinct physical objects are behind the images, and how many images come
> from each one?**

Everything else is logistics. This one determines whether the results are real.

If FFM is 1,200 images of 12 fiber classes, the number that governs statistical power
is not 1,200. It is the number of *physical fibers* photographed. A plausible
construction is 12 classes x 5 fibers x 20 crops = 1,200 images from **60 objects**.
In that case:

- Splitting by image puts crops of the *same fiber* in both train and test. The model
  recognises the fiber, not the fiber *type*, and accuracy goes up dramatically for a
  reason that has nothing to do with the method. This is the dominant failure mode in
  small forensic imaging studies and it is nearly invisible once it has happened.
- "1,200 images" also oversells the sample size by 20x in the manuscript.

The codebase already defends against this -- `group_disjoint` splitting partitions by
source object, and it is the default -- but it can only do so if each image carries its
specimen identity. Hence the concrete ask below.

---

## Concrete ask (minimum viable)

A directory tree, one folder per class, with the source object encoded in the filename:

```
FFM/
  cotton/     cotton_f01_c00.png  cotton_f01_c01.png  cotton_f02_c00.png ...
  polyester/  polyester_f01_c00.png ...
```

where `f01` identifies the physical fiber and `c00` the crop/field of view. Any
convention works as long as it is consistent -- the loader takes a regex
(`--group-pattern '(.+?)_c\d+'`). A separate CSV manifest mapping filename to specimen
id is equally fine.

If every image is genuinely a different physical object, say so explicitly and we drop
the grouping.

---

## Questions for Prof. Hemrajani

1. **Do the image sets exist as files today?** If so, who holds them and in what format?
   If they are still to be collected, we should know now, because that changes the
   schedule more than any modelling decision will.
2. **Specimen structure** -- per class, how many physical objects, and how many images
   per object? (The question above.)
3. **Actual per-class counts**, not approximate. "Approximately 1,200" cannot go in a
   paper.
4. **Acquisition conditions** -- magnification, illumination, sensor, resolution. The
   draft mentions polarised light for fibers and SEM for paint chips; confirming this is
   needed for the methods section regardless.
5. **Were classes imaged in separate sessions?** If class A was shot on one day and
   class B on another, session artifacts (lamp colour, focus, mounting) become a
   shortcut the classifier can exploit. If so, we need the session id per image and must
   report it as a limitation.
6. **Labelling procedure** -- who assigned class labels, on what criteria, and was any
   independent reference method (FTIR, SEM-EDX, refractive index) used to confirm them?
7. **Licensing and redistribution** -- can the data be released with the paper?
   Reviewers increasingly require it, and an unreleasable in-house dataset is the main
   reason this kind of paper gets rejected.

---

## If the datasets do not exist

Then we substitute, and the framing shifts from "three forensic datasets" to "a
forensic-adjacent microscopy benchmark plus a standard cross-domain benchmark." The
method is unchanged; only the application claim narrows.

| Candidate | Why | Status |
|---|---|---|
| **LUWA** (lithic use-wear, CVPR 2024) | Microscopic, genuinely few-shot, forensic-adjacent trace analysis. Already reference [2] in the professor's own SOS list, so it is defensible as "the dataset we said we would use." | Strongest option; needs manual download |
| **BSCD-FSL** (CropDisease, EuroSAT, ISIC, ChestX) | The standard cross-domain few-shot benchmark. Numbers directly comparable to published work. | EuroSAT already wired up and running |
| Materials/texture microscopy (NEU, MetalDAM, Kylberg) | Real domain shift from natural images, trivially available | Fallback only -- the forensic framing weakens |

The recommendation in `ARC_SPEC.md` section 2 stands either way: run BSCD-FSL alongside
whatever forensic data materialises. It is what makes the controller credible
independently of the application.
