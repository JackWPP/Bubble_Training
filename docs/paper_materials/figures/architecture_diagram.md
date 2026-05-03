# Bubble-YOLO11s Architecture

```mermaid
graph TB
    subgraph Input
        A[Input Image<br/>768×768×3]
    end

    subgraph Backbone["YOLO11s Backbone"]
        B1[Conv Stem<br/>Layer 0]
        B2[C2f Block<br/>Layer 2]
        B3[C2f Block<br/>Layer 4]
        B4[C2f Block<br/>Layer 6]
        B5[SPPF<br/>Layer 9]
    end

    subgraph Neck["YOLO11s Neck"]
        N1[Upsample + Concat<br/>Layer 12]
        N2[C2f<br/>Layer 13]
        N3[Upsample + Concat<br/>Layer 15]
        N4[C2f<br/>Layer 16]
        N5["★ P3LCRefine ★<br/>Layer 17<br/>(NEW)"]
        N6[Conv + Concat<br/>Layer 19]
        N7[C2f<br/>Layer 20]
        N8[Conv + Concat<br/>Layer 22]
        N9[C2f<br/>Layer 23]
    end

    subgraph Head["Detection Head"]
        H1[P3 Detect<br/>80×80]
        H2[P4 Detect<br/>40×40]
        H3[P5 Detect<br/>20×20]
    end

    subgraph Loss["Loss Function"]
        L1["CIoU Loss (95%)"]
        L2["NWD Loss (5%)"]
        L3["Box Loss = 0.95×CIoU + 0.05×NWD"]
    end

    A --> B1
    B1 --> B2 --> B3 --> B4 --> B5
    B4 -.->|skip| N1
    B3 -.->|skip| N3
    B5 --> N1
    N1 --> N2 --> N3
    N3 --> N4 --> N5
    N5 --> H1
    N5 --> N6 --> N7 --> N8 --> N9
    N2 --> N6
    N7 --> H2
    N9 --> H3
    H1 & H2 & H3 --> L1
    H1 & H2 & H3 --> L2
    L1 & L2 --> L3

    style N5 fill:#FF5722,color:white,stroke:#BF360C,stroke-width:3px
    style L2 fill:#2196F3,color:white
    style L3 fill:#4CAF50,color:white
```

## P3LCRefine Internal Structure

```mermaid
graph LR
    subgraph P3LCRefine
        X[Input x<br/>B×128×H×W] --> AP[AvgPool 3×3]
        AP --> SUB["contrast = x - avg"]
        X --> SUB
        SUB --> DW[DWConv 3×3]
        DW --> G["× γ (learnable)"]
        X --> ADD((+))
        G --> ADD
        ADD --> Y[Output y<br/>B×128×H×W]
    end
    
    style G fill:#FF5722,color:white
    style ADD fill:#4CAF50,color:white
```

## NWD Loss Flow

```mermaid
graph LR
    subgraph NWD
        PB[Pred Box<br/>x1,y1,x2,y2] --> G1[2D Gaussian<br/>N(μ, Σ)]
        TB[Target Box<br/>x1,y1,x2,y2] --> G2[2D Gaussian<br/>N(μ, Σ)]
        G1 --> W[Wasserstein Distance]
        G2 --> W
        W --> E["exp(-W²/C)"]
        E --> NWD[NWD Similarity]
        NWD --> LOSS["L_NWD = 1 - NWD"]
    end
    
    subgraph Fusion
        CIOU[CIoU Loss] --> BL["Box Loss<br/>= 0.95×CIoU<br/>+ 0.05×NWD"]
        LOSS --> BL
    end
    
    style W fill:#2196F3,color:white
    style BL fill:#4CAF50,color:white
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Total Parameters | 9,414,340 |
| P3LCRefine Parameters | ~1,280 (0.013%) |
| FLOPs | 21.3G |
| Inference Time (V100) | ~4.0 ms/img |
| Input Resolution | 768 × 768 |
| Number of Classes | 1 (bubble) |
