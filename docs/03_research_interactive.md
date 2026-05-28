# Research: Interactive Teaching Methods for Deep Learning (HTML-Based Course Materials)

> Compiled 2026-05-28. Focused on practical, lightweight implementations using vanilla JS + minimal libraries.

---

## Table of Contents

1. [Best Practices for Visual/Interactive Deep Learning Teaching](#1-best-practices-for-visualinteractive-deep-learning-teaching)
2. [JavaScript Libraries for Neural Network Visualization](#2-javascript-libraries-for-neural-network-visualization)
3. [Animation Libraries for Educational Content](#3-animation-libraries-for-educational-content)
4. [Lightweight Interactive HTML/CSS Frameworks](#4-lightweight-interactive-htmlcss-frameworks)
5. [Exemplary Interactive Deep Learning Websites](#5-exemplary-interactive-deep-learning-websites)
6. [Embedding Interactive Demos in HTML](#6-embedding-interactive-demos-in-html)
7. [CSS Frameworks for Course Pages](#7-css-frameworks-for-course-pages)
8. [Math Rendering in HTML](#8-math-rendering-in-html)
9. [Embeddable Code Editors](#9-embeddable-code-editors)
10. [Color Schemes and Typography for Technical Education](#10-color-schemes-and-typography-for-technical-education)
11. [Recommended Tech Stack Summary](#11-recommended-tech-stack-summary)

---

## 1. Best Practices for Visual/Interactive Deep Learning Teaching

### Pedagogical Principles

**Progressive disclosure.** Start with simple concepts and layer complexity. TensorFlow Playground exemplifies this -- users begin with a single-layer network and can add layers/neurons incrementally.

**Immediate feedback loops.** Every parameter change should produce a visible result. The user adjusts a slider, the visualization updates in real time. No submit buttons, no waiting.

**Multiple representations.** Show the same concept through different lenses simultaneously:
- Mathematical notation (equations)
- Geometric/visual representation (diagrams, heatmaps)
- Code implementation (executable snippets)
- Intuitive analogy (everyday examples)

**Scaffolded complexity.** Follow the pattern used by Distill.pub and 3Blue1Brown:
1. Start with intuition (visual, geometric)
2. Formalize with math
3. Connect to implementation
4. Let the reader experiment

**Color as information.** TensorFlow Playground uses orange/blue to encode negative/positive values. CNN Explainer uses color intensity to show activation magnitudes. Consistent color encoding across a course builds transferable intuition.

**"Play first, explain later."** Let users interact with a demo before reading the explanation. This creates curiosity and a mental model that formal definitions can attach to.

### Content Architecture for a UNet Course

```
Page flow per topic:
  1. Interactive demo (hands-on, no explanation yet)
  2. "What just happened?" -- brief intuitive explanation
  3. Mathematical formalization (with KaTeX)
  4. Code walkthrough (with embedded editor)
  5. Exercises (modify the demo, answer questions)
```

---

## 2. JavaScript Libraries for Neural Network Visualization

### Tier 1: Recommended for This Project

| Library | Size | Best For | CDN Available |
|---------|------|----------|---------------|
| **TensorFlow.js** | ~150KB (min+gz) | Running real models in-browser, inference demos | Yes |
| **D3.js** | ~90KB | Custom network diagrams, data visualizations, SVG manipulation | Yes |
| **brain.js** | ~50KB | Simple NN demos, beginner-friendly API | Yes |

### Tier 2: Useful Supplementary

| Library | Size | Best For | CDN Available |
|---------|------|----------|---------------|
| **ConvNetJS** | ~25KB | CNN demos (created by Andrej Karpathy) | GitHub only |
| **ONNX Runtime Web** | ~200KB | Running pre-trained ONNX models in browser | Yes |
| **ml5.js** | ~100KB | Friendly wrapper around TensorFlow.js | Yes |

### Tier 3: Specialized

| Library | Size | Best For | CDN Available |
|---------|------|----------|---------------|
| **TensorSpace.js** | ~50KB | 3D neural network architecture visualization (uses Three.js) | Yes |
| **Netron** | N/A (web app) | Viewing model architecture files (.onnx, .pb, .h5) | Web app |
| **neataptic** | ~30KB | Neuroevolution visualization | GitHub |

### Practical Recommendations

**For the UNet course, use this combination:**

1. **D3.js** for all custom diagrams (architecture diagrams, feature map visualizations, convolution animations). It gives full control over SVG/Canvas rendering and is the industry standard for data visualization.

2. **TensorFlow.js** only when you need to run actual inference in the browser (e.g., "upload an image and see UNet segment it live"). Do NOT use it for teaching concepts -- it is too heavy for simple visualizations.

3. **Vanilla Canvas API** for simple animations (data flowing through layers, convolution kernels sliding across images). No library needed for these.

4. **Custom nn.ts** (inspired by TensorFlow Playground) for lightweight educational neural network simulations. TensorFlow Playground itself uses a custom ~200-line neural network library, not TensorFlow.js.

### Code Example: Minimal NN Visualization with Canvas

```html
<canvas id="nn-canvas" width="800" height="400"></canvas>
<script>
const canvas = document.getElementById('nn-canvas');
const ctx = canvas.getContext('2d');

// Define network architecture
const layers = [
  { neurons: 3, label: 'Input' },
  { neurons: 5, label: 'Conv 1' },
  { neurons: 5, label: 'Conv 2' },
  { neurons: 3, label: 'Output' }
];

function drawNetwork() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const layerSpacing = canvas.width / (layers.length + 1);

  layers.forEach((layer, li) => {
    const x = layerSpacing * (li + 1);
    const neuronSpacing = canvas.height / (layer.neurons + 1);

    // Draw connections to next layer
    if (li < layers.length - 1) {
      const nextLayer = layers[li + 1];
      const nextX = layerSpacing * (li + 2);
      for (let i = 1; i <= layer.neurons; i++) {
        for (let j = 1; j <= nextLayer.neurons; j++) {
          ctx.beginPath();
          ctx.moveTo(x, neuronSpacing * i);
          ctx.lineTo(nextX, (canvas.height / (nextLayer.neurons + 1)) * j);
          ctx.strokeStyle = 'rgba(100, 149, 237, 0.3)';
          ctx.stroke();
        }
      }
    }

    // Draw neurons
    for (let i = 1; i <= layer.neurons; i++) {
      ctx.beginPath();
      ctx.arc(x, neuronSpacing * i, 12, 0, Math.PI * 2);
      ctx.fillStyle = '#4a90d9';
      ctx.fill();
      ctx.strokeStyle = '#2c5f8a';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw labels
    ctx.fillStyle = '#333';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(layer.label, x, canvas.height - 10);
  });
}

drawNetwork();
</script>
```

---

## 3. Animation Libraries for Educational Content

### Comparison Matrix

| Library | Size | Learning Curve | Best For | Scroll Support | CDN |
|---------|------|----------------|----------|----------------|-----|
| **GSAP** | ~30KB | Moderate | Timeline-based animations, scroll-triggered effects | Yes (ScrollTrigger) | Yes |
| **anime.js** | ~17KB | Low | Simple property animations, lightweight alternative to GSAP | No (manual) | Yes |
| **D3.js** | ~90KB | High | Data-driven transitions, SVG/Canvas animations | No (manual) | Yes |
| **p5.js** | ~300KB | Low | Creative coding, interactive sketches, simulations | No (manual) | Yes |
| **Three.js** | ~150KB | High | 3D visualizations | No (manual) | Yes |
| **Lottie** | ~50KB | Low | After Effects animations exported as JSON | No | Yes |
| **Vanilla Canvas** | 0KB | Moderate | Custom animations, full control | N/A | N/A |

### Recommendation for Course Content

**Primary: Vanilla Canvas + requestAnimationFrame**

For educational deep learning content, you mostly need:
- Convolution kernels sliding over feature maps
- Data flowing through network layers
- Gradient values propagating backward
- Pixel-level color changes (showing activations)

All of these are straightforward with the Canvas API. No library overhead, full control, zero dependencies.

**Secondary: GSAP with ScrollTrigger (free for non-commercial)**

Use GSAP for:
- Scroll-driven narrative ("scrollytelling") sections
- Smooth transitions between explanation sections
- Entrance/exit animations for diagrams
- Complex timeline sequences that would be tedious in vanilla JS

**Tertiary: anime.js**

If you need lightweight property animations but GSAP feels like overkill:
- Animating SVG attributes (stroke-dashoffset for drawing effects)
- Simple easing on opacity/transform
- Staggered animations across multiple elements

**Avoid for this project:**
- **Three.js** -- overkill unless you specifically need 3D UNet visualization
- **Lottie** -- designed for pre-made After Effects animations, not programmatic educational content
- **p5.js** -- great for creative coding courses but adds too much weight for what we need

### Scrollytelling Pattern (GSAP + ScrollTrigger)

```html
<div class="explanation-section" id="conv-demo">
  <div class="sticky-visualization">
    <canvas id="conv-canvas"></canvas>
  </div>
  <div class="scroll-text">
    <div class="step" data-step="1">
      <h3>Step 1: The Input Image</h3>
      <p>We start with a grayscale image represented as a 2D matrix...</p>
    </div>
    <div class="step" data-step="2">
      <h3>Step 2: The Kernel</h3>
      <p>A 3x3 kernel slides across the image...</p>
    </div>
    <div class="step" data-step="3">
      <h3>Step 3: Element-wise Multiplication</h3>
      <p>At each position, we multiply and sum...</p>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script>
gsap.registerPlugin(ScrollTrigger);

document.querySelectorAll('.step').forEach((step, i) => {
  ScrollTrigger.create({
    trigger: step,
    start: 'top center',
    end: 'bottom center',
    onEnter: () => updateConvVisualization(i),
    onEnterBack: () => updateConvVisualization(i),
  });
});
</script>
```

---

## 4. Lightweight Interactive HTML/CSS Frameworks

### Alpine.js (Recommended for Interactivity)

**What it is:** A lightweight (~15KB) framework that adds declarative interactivity directly in HTML markup.

**Why it fits:** No build step required. Drop in a `<script>` tag and use `x-data`, `x-show`, `x-on`, `x-model` directives. Co-locates behavior with markup.

**Use cases for the course:**
- Toggle between different visualization modes
- Show/hide mathematical derivations
- Tab switching between "intuition / math / code" views
- Interactive quizzes with immediate feedback
- Collapsible sections for advanced content

```html
<!-- Example: Toggle between visualization modes -->
<div x-data="{ mode: 'forward' }">
  <button @click="mode = 'forward'" :class="{ active: mode === 'forward' }">
    Forward Pass
  </button>
  <button @click="mode = 'backward'" :class="{ active: mode === 'backward' }">
    Backward Pass
  </button>

  <div x-show="mode === 'forward'" x-transition>
    <!-- Forward pass visualization -->
  </div>
  <div x-show="mode === 'backward'" x-transition>
    <!-- Backward pass visualization -->
  </div>
</div>
```

### Other Options Considered

| Framework | Size | Build Step | Verdict |
|-----------|------|------------|---------|
| **Alpine.js** | ~15KB | No | **Best fit** -- declarative, no build, CDN-ready |
| **Stimulus** | ~20KB | Optional | Good but requires understanding of controllers |
| **Lit** | ~15KB | No | Web Components focused, more boilerplate |
| **htmx** | ~14KB | No | Good for server-driven apps, less relevant here |
| **Vue.js** | ~40KB | No (CDN) | Overkill for static course pages |
| **React** | ~45KB | Yes | Way overkill, requires build toolchain |

### Vanilla JS Patterns (When Alpine Is Overkill)

For simple interactions, vanilla JS is perfectly fine:

```javascript
// Toggle visibility
document.querySelectorAll('[data-toggle]').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = document.querySelector(btn.dataset.toggle);
    target.hidden = !target.hidden;
  });
});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn, .tab-panel').forEach(el =>
      el.classList.remove('active')
    );
    btn.classList.add('active');
    document.querySelector(btn.dataset.tab).classList.add('active');
  });
});
```

---

## 5. Exemplary Interactive Deep Learning Websites

### Analysis of Best-in-Class Examples

#### TensorFlow Playground (playground.tensorflow.org)

**Technologies:** Custom TypeScript neural network library (`nn.ts`), D3.js for SVG rendering, custom CSS.

**Key design patterns:**
- Real-time parameter adjustment with immediate visual feedback
- Color encoding: blue = positive, orange = negative
- Line thickness encodes weight magnitude
- Background color intensity shows prediction confidence
- Play/pause controls for watching training unfold
- Configurable architecture (add/remove layers and neurons)
- Dataset selection with adjustable noise levels

**Takeaway:** Build a custom, minimal NN library rather than pulling in TensorFlow.js. The Playground's `nn.ts` is ~200 lines and does everything needed for education.

#### Distill.pub

**Technologies:** Custom CSS grid system, Prism.js for code, custom hover/citation system, per-article custom JS (D3.js, TensorFlow.js, Canvas API).

**Key design patterns:**
- Responsive column layout: `l-body`, `l-middle`, `l-page`, `l-screen`
- Grid: 60px column, 24px gutter, 648px body width
- Interactive figures that respond to user input (sliders, hover, drag)
- Annotations directly on visualizations
- Peer-reviewed quality with interactive exploration
- Citation hover boxes with 250ms dismiss timeout

**Takeaway:** The Distill article format (wide interactive figures alongside flowing text) is the gold standard for ML educational content. Replicate this layout pattern.

#### CNN Explainer (poloclub.github.io/cnn-explainer)

**Technologies:** Svelte, D3.js, TensorFlow.js (for running actual CNN inference), custom SVG visualizations.

**Key design patterns:**
- Animated data flow through network layers
- Interactive exploration of individual layer outputs
- Click on any layer to see its feature maps
- Smooth zoom transitions between overview and detail
- Color-coded activation maps

**Takeaway:** The "click any layer to explore" pattern is powerful for understanding UNet's encoder-decoder structure.

#### Andrej Karpathy's ConvNetJS Demos

**Technologies:** ConvNetJS (custom library), Canvas API for visualization.

**Key design patterns:**
- Live training visualization
- Multiple demo presets (regression, classification, etc.)
- Minimal UI -- the visualization IS the interface

### Other Notable Resources

| Resource | URL | What to Learn From It |
|----------|-----|----------------------|
| **3Blue1Brown (Manim)** | 3blue1brown.com | Mathematical animation style, geometric intuition |
| **Lil'Log (Lilian Weng)** | lilianweng.github.io | Blog post structure, comprehensive yet readable |
| **Jay Alammar's Blog** | jalammar.github.io | Beautiful static visualizations of transformers |
| **Bert Hubert's "Transformer in C"** | github.com/bert-huiber | Explaining transformers with minimal code |
| **Machine Learning for Beginners** | github.com/microsoft/ML-For-Beginners | Structured curriculum with interactive notebooks |

---

## 6. Embedding Interactive Demos in HTML

### Option A: Inline Canvas/SVG (Recommended for Most Cases)

Best for: Custom visualizations that are tightly coupled with the page content.

```html
<div class="demo-container">
  <canvas id="demo" width="600" height="400"></canvas>
  <div class="controls">
    <label>Learning Rate: <input type="range" id="lr" min="0.001" max="1" step="0.001" value="0.01"></label>
    <label>Epochs: <input type="range" id="epochs" min="1" max="100" value="10"></label>
    <button id="train-btn">Train</button>
    <button id="reset-btn">Reset</button>
  </div>
</div>
<script src="demo.js"></script>
```

**Pros:** Full control, no iframe overhead, shares page styles, easy to communicate with other page elements.
**Cons:** Can conflict with page CSS/JS if not namespaced carefully.

### Option B: Web Components (Recommended for Reusable Demos)

Best for: Demos that appear on multiple pages or need style isolation.

```html
<script>
class ConvDemo extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    shadow.innerHTML = `
      <style>
        /* Scoped styles -- no leakage */
        :host { display: block; margin: 2rem 0; }
        canvas { border: 1px solid #ddd; border-radius: 8px; }
        .controls { margin-top: 1rem; }
      </style>
      <canvas width="600" height="400"></canvas>
      <div class="controls">
        <slot name="controls"></slot>
      </div>
    `;
  }

  connectedCallback() {
    const canvas = this.shadowRoot.querySelector('canvas');
    this.ctx = canvas.getContext('2d');
    this.initDemo();
  }

  initDemo() {
    // Demo logic here
  }
}

customElements.define('conv-demo', ConvDemo);
</script>

<!-- Usage -->
<conv-demo>
  <div slot="controls">
    <input type="range" min="1" max="10" value="3">
  </div>
</conv-demo>
```

**Pros:** Full style isolation via Shadow DOM, reusable across pages, clean API.
**Cons:** More boilerplate, slot-based content injection can be awkward.

### Option C: Iframes (Use Sparingly)

Best for: Third-party content, completely independent demos, security-sensitive code.

```html
<iframe
  src="demos/convolution.html"
  title="Interactive Convolution Demo"
  width="100%"
  height="500"
  loading="lazy"
  sandbox="allow-scripts"
  style="border: 1px solid #ddd; border-radius: 8px;"
></iframe>
```

**Pros:** Complete isolation, can embed external content.
**Cons:** Communication requires `postMessage`, layout sizing is tricky, extra HTTP request, poor SEO.

### Option D: CodePen/JSFiddle Embeds (Quick Prototyping)

Best for: Prototyping, showing community contributions, external examples.

```html
<p class="codepen" data-height="400" data-theme-id="light" data-default-tab="result"
   data-slug-hash="abc123" data-user="username">
  <span>See the Pen <a href="https://codepen.io/username/pen/abc123">Convolution Demo</a></span>
</p>
<script async src="https://cpwebassets.codepen.io/assets/embed/ei.js"></script>
```

**Verdict:** Good for prototyping. For a polished course, migrate to self-hosted demos.

### Recommended Approach

| Demo Type | Embedding Method |
|-----------|-----------------|
| Architecture diagrams | Inline SVG (D3.js) |
| Training visualizations | Inline Canvas |
| Layer explorers | Inline Canvas + Alpine.js controls |
| Standalone mini-apps | Web Components |
| External/third-party | Iframes |
| Quick prototypes | CodePen embeds |

---

## 7. CSS Frameworks for Course Pages

### Tier 1: Strongly Recommended

#### Pico CSS

**Size:** ~10KB (classless version even smaller)
**Philosophy:** Style semantic HTML directly, no classes needed.

**Why it fits:**
- Automatic dark/light mode via `prefers-color-scheme`
- Responsive by default (font sizes scale with viewport)
- 130+ CSS custom properties for customization
- No JavaScript dependency
- Works with plain `<article>`, `<section>`, `<nav>`, `<figure>` elements

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@latest/css/pico.min.css">

<!-- Just write semantic HTML and it looks good -->
<main class="container">
  <article>
    <header><h1>Convolutional Layers</h1></header>
    <p>The convolution operation slides a kernel across the input...</p>
    <figure>
      <canvas id="demo"></canvas>
      <figcaption>Interactive convolution demo</figcaption>
    </figure>
  </article>
</main>
```

#### Simple.css

**Size:** ~10KB
**Philosophy:** Mostly classless, designed for simple content sites.

**Why it fits:**
- Automatic dark mode
- Good typography out of the box
- Extremely lightweight
- Perfect for blog-style course content
- MIT licensed

### Tier 2: Good Alternatives

#### Tufte CSS

**Philosophy:** Inspired by Edward Tufte's book design. Excellent for dense technical content with margin notes.

**Key features:**
- Margin notes for supplementary explanations
- Sidenotes instead of footnotes
- Clean typography with `et-book` font
- Minimal decoration, maximum content density

**Use for:** Mathematical derivations, dense reference sections.

#### Water.css

**Size:** ~2KB
**Philosophy:** Drop-in styles for simple HTML pages. No classes.

### Recommendation

**Use Pico CSS as the base.** It handles 90% of what you need:
- Clean typography
- Responsive layout
- Dark mode
- Form styling (for interactive controls)
- Table styling (for comparison tables)
- Code block styling

Then add a small custom stylesheet (50-100 lines) for:
- Course-specific colors
- Demo container styling
- Custom interactive element styling
- Distill-style wide figures

```css
/* Custom overrides on top of Pico */
:root {
  --primary: #2563eb;
  --primary-light: #60a5fa;
  --surface: #f8fafc;
  --text: #1e293b;
}

@media (prefers-color-scheme: dark) {
  :root {
    --primary: #60a5fa;
    --primary-light: #93c5fd;
    --surface: #0f172a;
    --text: #e2e8f0;
  }
}

.demo-container {
  background: var(--surface);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 2rem 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.wide-figure {
  width: 100vw;
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  padding: 2rem;
}
```

---

## 8. Math Rendering in HTML

### KaTeX (Recommended)

**What it is:** A fast math typesetting library by Khan Academy.

**Why it fits:**
- Renders ~100x faster than MathJax v2
- Synchronous rendering (no layout jumps)
- Smaller bundle size (~300KB vs MathJax's ~500KB+)
- Works well with dynamic content (re-render on state change)
- MIT licensed

**Key API:**

```javascript
// Render to a DOM element
katex.render("E = mc^2", element, { throwOnError: false });

// Render to HTML string
const html = katex.renderToString("\\frac{\\partial L}{\\partial w} = \\frac{1}{N} \\sum_{i=1}^{N} (\\hat{y}_i - y_i) x_i", {
  throwOnError: false,
  displayMode: true
});

// Persistent macros across renders
const macros = { "\\R": "\\mathbb{R}" };
katex.render("\\vec{x} \\in \\R^n", el, { macros });
```

**Auto-render extension** (detects and renders all math on a page):

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false},
      {left: '\\[', right: '\\]', display: true},
      {left: '\\(', right: '\\)', display: false}
    ]
  });"></script>
```

**Common UNet/course LaTeX snippets:**

```
% Cross-entropy loss
L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]

% Dice coefficient
D = \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}

% Convolution operation
(f * g)(x, y) = \sum_{m} \sum_{n} f(m, n) \cdot g(x - m, y - n)

% UNet skip connection
x_l = \text{concat}\left( \text{upsample}(x_{l+1}),\ \text{crop}(x_{L-l}) \right)
```

### MathJax v3 (Alternative)

**When to prefer MathJax over KaTeX:**
- You need accessibility features (MathML output, screen reader support)
- You need broader LaTeX command support
- You need SVG output (KaTeX outputs HTML/CSS)

**Key API (MathJax v3):**

```html
<script>
MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
```

### Decision Matrix

| Feature | KaTeX | MathJax v3 |
|---------|-------|------------|
| Render speed | Very fast | Moderate |
| Bundle size | ~300KB | ~500KB+ |
| LaTeX coverage | ~95% | ~99% |
| Accessibility | Basic | Excellent (MathML) |
| SVG output | No | Yes |
| Auto-render | Yes (extension) | Yes (built-in) |
| Dynamic content | Excellent | Good |
| **Recommendation** | **Use this** | Fallback if needed |

---

## 9. Embeddable Code Editors

### CodeMirror 6 (Recommended)

**What it is:** A modular, extensible code editor for the browser.

**Why it fits:**
- ~100KB for a basic setup (modular, import only what you need)
- Excellent syntax highlighting (Python, JavaScript, etc.)
- Line numbers, code folding, bracket matching
- Mobile-friendly
- MIT licensed
- Active development, modern architecture

**Basic setup:**

```html
<script type="module">
import { EditorView, basicSetup } from 'https://cdn.jsdelivr.net/npm/@codemirror/basic-setup/+esm';
import { python } from 'https://cdn.jsdelivr.net/npm/@codemirror/lang-python/+esm';
import { oneDark } from 'https://cdn.jsdelivr.net/npm/@codemirror/theme-one-dark/+esm';

const editor = new EditorView({
  doc: `import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)`,
  extensions: [basicSetup, python(), oneDark],
  parent: document.getElementById('editor-container')
});
</script>

<div id="editor-container" style="border-radius: 8px; overflow: hidden;"></div>
```

### Monaco Editor (VS Code's Editor)

**When to use:** If you need IntelliSense, minimap, or a full IDE experience.

**Why NOT to use for this project:**
- ~2MB bundle size (heavier than the entire rest of the course combined)
- Overkill for displaying code snippets
- Complex setup

**Verdict:** Avoid. Use CodeMirror for editable code, `<pre><code>` with Prism.js/Highlight.js for read-only.

### Read-Only Code Display

For non-editable code blocks, use **Prism.js** (~10KB):

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>

<pre><code class="language-python">
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
</code></pre>
```

### Recommendation

| Use Case | Tool |
|----------|------|
| Editable code blocks (exercises) | CodeMirror 6 |
| Read-only code display | Prism.js |
| "Try it yourself" Python execution | Pyodide (in-browser Python) or iframe to Replit |

---

## 10. Color Schemes and Typography for Technical Educational Content

### Color Palette Recommendations

#### Light Mode (Primary)

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background | White | `#ffffff` | Page background |
| Surface | Light gray | `#f8fafc` | Cards, demo containers |
| Text primary | Dark slate | `#1e293b` | Body text |
| Text secondary | Medium gray | `#64748b` | Captions, metadata |
| Primary accent | Blue | `#2563eb` | Links, buttons, highlights |
| Secondary accent | Teal | `#0d9488` | Success states, positive values |
| Warning | Amber | `#d97706` | Warnings, attention |
| Error | Red | `#dc2626` | Errors, negative values |
| Code background | Light gray | `#f1f5f9` | Inline code, code blocks |

#### Dark Mode

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background | Deep navy | `#0f172a` | Page background |
| Surface | Dark slate | `#1e293b` | Cards, demo containers |
| Text primary | Light gray | `#e2e8f0` | Body text |
| Text secondary | Medium gray | `#94a3b8` | Captions, metadata |
| Primary accent | Light blue | `#60a5fa` | Links, buttons, highlights |
| Secondary accent | Teal | `#2dd4bf` | Success, positive values |

#### Semantic Colors for NN Visualizations

| Concept | Color | Hex |
|---------|-------|-----|
| Positive values / activations | Blue | `#3b82f6` |
| Negative values | Orange | `#f97316` |
| Weights (large magnitude) | Dark | `#1e293b` |
| Weights (small magnitude) | Light | `#cbd5e1` |
| Gradient flow | Green to Red | `#22c55e` to `#ef4444` |
| Input layer | Sky blue | `#38bdf8` |
| Hidden layers | Indigo | `#6366f1` |
| Output layer | Violet | `#8b5cf6` |
| Skip connections | Amber | `#f59e0b` |

### Typography Recommendations

#### Font Stack for Body Text

```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
```

**Inter** is the best choice for technical content:
- Designed specifically for screens
- Excellent readability at small sizes
- Tabular numbers for code/data
- Free and open source (OFL license)
- Available on Google Fonts

#### Font Stack for Code

```css
font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
```

**JetBrains Mono** is recommended:
- Designed for reading code
- Excellent ligatures (optional)
- Good distinction between similar characters (0/O, 1/l/I)
- Free and open source

#### Typographic Scale

```css
:root {
  --text-xs: 0.75rem;    /* 12px -- captions, labels */
  --text-sm: 0.875rem;   /* 14px -- code, small text */
  --text-base: 1rem;     /* 16px -- body text */
  --text-lg: 1.125rem;   /* 18px -- lead paragraphs */
  --text-xl: 1.25rem;    /* 20px -- h4 */
  --text-2xl: 1.5rem;    /* 24px -- h3 */
  --text-3xl: 1.875rem;  /* 30px -- h2 */
  --text-4xl: 2.25rem;   /* 36px -- h1 */

  --line-height: 1.7;    /* Generous for readability */
  --line-height-tight: 1.3;  /* For headings */
  --max-width: 680px;    /* Optimal reading width */
  --max-width-wide: 960px;  /* Wide content (figures) */
}
```

#### Reading Experience Best Practices

- **Line length:** 60-75 characters per line (use `max-width: 680px`)
- **Line height:** 1.6-1.8 for body text, 1.2-1.3 for headings
- **Paragraph spacing:** 1.5em margin between paragraphs
- **Font size:** Minimum 16px for body text
- **Code blocks:** Slightly smaller (14px) with more line spacing (1.5)
- **Math:** Render at the same size as surrounding text for inline, slightly larger for display mode

### Dark Mode Implementation

```css
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --surface: #1e293b;
    --text: #e2e8f0;
    --text-secondary: #94a3b8;
    --primary: #60a5fa;
    --border: #334155;
  }

  img, .demo-canvas {
    filter: brightness(0.9);
  }
}
```

---

## 11. Recommended Tech Stack Summary

### Minimal Stack (Zero Build Step)

```
HTML5 + CSS3 + Vanilla JavaScript
|
+-- Layout:        Pico CSS (CDN, ~10KB)
+-- Interactivity: Alpine.js (CDN, ~15KB)
+-- Math:          KaTeX (CDN, ~300KB)
+-- Code Display:  Prism.js (CDN, ~10KB)
+-- Diagrams:      D3.js (CDN, ~90KB) or Vanilla Canvas/SVG
+-- Animation:     Vanilla requestAnimationFrame + GSAP ScrollTrigger (CDN, ~30KB)
+-- Fonts:         Inter + JetBrains Mono (Google Fonts)
```

**Total added weight: ~455KB** (mostly KaTeX). Everything else is under 150KB combined.

### File Structure

```
course/
  index.html
  css/
    pico.min.css        (CDN link, not local)
    custom.css          (~100 lines of overrides)
  js/
    main.js             (page-level initialization)
    demos/
      convolution.js    (convolution animation)
      unet-arch.js      (UNet architecture diagram)
      skip-connections.js
      feature-maps.js
      loss-landscape.js
  chapters/
    01-introduction.html
    02-convolution.html
    03-pooling.html
    ...
    10-unet.html
```

### CDN Links (Copy-Paste Ready)

```html
<!-- Pico CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false}
    ]
  });"></script>

<!-- Alpine.js -->
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>

<!-- Prism.js (code highlighting) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.min.css">
<script src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-python.min.js"></script>

<!-- GSAP (only if using scroll animations) -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>

<!-- D3.js (only on pages that need it) -->
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>

<!-- Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

### Template HTML Page

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chapter 5: Convolutional Layers -- UNet Course</title>

  <!-- CSS -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
  <link rel="stylesheet" href="css/custom.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">

  <style>
    body { font-family: 'Inter', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .demo-container {
      background: var(--pico-card-background-color);
      border-radius: 12px;
      padding: 1.5rem;
      margin: 2rem 0;
    }
  </style>
</head>
<body>
  <main class="container">
    <nav>
      <a href="/">UNet Course</a>
      <ul>
        <li><a href="/chapters/04.html">Previous</a></li>
        <li><a href="/chapters/06.html">Next</a></li>
      </ul>
    </nav>

    <article>
      <header>
        <h1>Chapter 5: Convolutional Layers</h1>
        <p>Understanding the fundamental building block of CNNs</p>
      </header>

      <!-- Interactive Demo (before explanation) -->
      <div class="demo-container" id="conv-demo">
        <canvas id="conv-canvas" width="700" height="400"></canvas>
        <div class="controls">
          <label>Kernel Size:
            <input type="range" min="2" max="7" value="3" id="kernel-size">
          </label>
          <label>Stride:
            <input type="range" min="1" max="4" value="1" id="stride">
          </label>
        </div>
      </div>

      <!-- Explanation -->
      <h2>What Just Happened?</h2>
      <p>The convolution operation slides a small matrix (the <em>kernel</em>)
         across the input image. At each position, we compute...</p>

      <!-- Math -->
      <h2>The Mathematics</h2>
      <p>The convolution of input $I$ with kernel $K$ is defined as:</p>
      $$$(I * K)(x, y) = \sum_{m=-k}^{k} \sum_{n=-k}^{k} I(x+m, y+n) \cdot K(m, n)$$$

      <!-- Code -->
      <h2>In PyTorch</h2>
      <pre><code class="language-python">conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=1
)</code></pre>

    </article>
  </main>

  <!-- Scripts -->
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {
      delimiters: [
        {left: '$$$', right: '$$$', display: true},
        {left: '$', right: '$', display: false}
      ]
    });"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>
  <script src="js/demos/convolution.js"></script>
</body>
</html>
```

---

## Key Takeaways

1. **Keep it vanilla.** Avoid React, Vue, or any framework that requires a build step. HTML + CSS + vanilla JS + a few CDN libraries is all you need.

2. **KaTeX over MathJax.** 100x faster rendering matters when you have many equations per page.

3. **Pico CSS for the base, custom CSS for personality.** Don't write a CSS framework from scratch.

4. **Canvas API for demos.** Most deep learning visualizations (convolutions, feature maps, data flow) map naturally to pixel-level Canvas operations.

5. **Alpine.js for UI state.** Tabs, toggles, collapsible sections, mode switching -- Alpine handles this cleanly without a build step.

6. **Custom NN library over TensorFlow.js.** Unless you need to run actual inference in the browser, a ~200-line custom neural network library (like TensorFlow Playground's `nn.ts`) is better for education.

7. **Distill.pub layout as your template.** Wide interactive figures alongside flowing text. Column-based responsive grid. Annotations on visualizations.

8. **Color encodes information.** Pick a consistent color scheme (blue=positive, orange=negative) and use it everywhere.

9. **Inter + JetBrains Mono.** Best-in-class fonts for technical content. Free. Available via Google Fonts.

10. **Progressive enhancement.** Every page should be readable without JavaScript. Demos enhance understanding but aren't required for comprehension.
