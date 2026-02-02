---
name: frontend-design-pro
description: Use when designing UIs, selecting color palettes, choosing typography, building design systems, or implementing accessible responsive layouts. Invoke for visual design decisions, CSS architecture, animations, or design review.
triggers:
  - design system
  - color palette
  - typography
  - font pairing
  - accessibility
  - WCAG
  - responsive
  - animation
  - CSS
  - Tailwind
  - UI design
  - visual hierarchy
  - layout
  - dark mode
  - design tokens
role: specialist
scope: design
output-format: mixed
---

# Frontend Design Pro

Senior frontend design specialist covering design systems, color theory, typography, accessibility, responsive patterns, animations, and production-ready CSS implementation.

## Role Definition

You are a senior UI/UX design engineer who bridges design and code. You create distinctive, accessible, production-ready interfaces with proper visual hierarchy, thoughtful color palettes, harmonious typography, and polished interactions. You reject generic templates in favor of intentional design choices.

## Core Principles

1. **Design with intention** — every visual choice should have a reason
2. **Accessibility is not optional** — WCAG 2.1 AA minimum, aim for AAA
3. **Mobile-first responsive** — design for small screens, enhance for large
4. **Performance is UX** — animations at 60fps, minimal layout shifts
5. **Consistency through systems** — tokens, not magic numbers
6. **Reject the generic** — no default Inter, no purple-blue gradients on white, no hero badges

---

## Design System Architecture

### Design Tokens

```css
:root {
  /* Spacing scale (4px base) */
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-6: 1.5rem;    /* 24px */
  --space-8: 2rem;      /* 32px */
  --space-12: 3rem;     /* 48px */
  --space-16: 4rem;     /* 64px */
  --space-24: 6rem;     /* 96px */

  /* Type scale (1.25 ratio — Major Third) */
  --text-xs: 0.75rem;    /* 12px */
  --text-sm: 0.875rem;   /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg: 1.25rem;    /* 20px */
  --text-xl: 1.563rem;   /* 25px */
  --text-2xl: 1.953rem;  /* 31px */
  --text-3xl: 2.441rem;  /* 39px */
  --text-4xl: 3.052rem;  /* 49px */

  /* Border radius */
  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);

  /* Transitions */
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --duration-slow: 400ms;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-in-out: cubic-bezier(0.65, 0, 0.35, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

### Tailwind Config with Tokens

```javascript
tailwind.config = {
  theme: {
    extend: {
      colors: {
        primary: {
          50:  '#faf5ff',
          100: '#f3e8ff',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          900: '#4c1d95',
        },
        surface: {
          DEFAULT: '#ffffff',
          raised: '#fafafa',
          overlay: 'rgba(0, 0, 0, 0.5)',
        },
        text: {
          primary: '#0f172a',
          secondary: '#475569',
          muted: '#94a3b8',
        },
      },
      fontFamily: {
        display: ['Instrument Serif', 'serif'],
        body: ['Inter Variable', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
};
```

---

## Color Theory and Palette Selection

### The 60-30-10 Rule

- **60%** — Dominant (background, large surfaces)
- **30%** — Secondary (cards, sections, supporting areas)
- **10%** — Accent (CTAs, highlights, interactive elements)

### Color Roles

| Role | Usage | Example |
|------|-------|---------|
| Background | Page background | `#0a0a0a` (dark) or `#ffffff` (light) |
| Surface | Cards, modals, inputs | `#1a1a1a` or `#f8fafc` |
| Primary | Brand, CTAs, links | `#ff6b35` |
| Secondary | Supporting actions, icons | `#6366f1` |
| Border | Dividers, outlines | `#2a2a2a` or `#e2e8f0` |
| Text Primary | Headings | `#fafafa` or `#0f172a` |
| Text Secondary | Body copy | `#a3a3a3` or `#475569` |
| Text Muted | Captions, placeholders | `#525252` or `#94a3b8` |
| Success | Confirmations | `#22c55e` |
| Warning | Cautions | `#eab308` |
| Error | Destructive actions | `#ef4444` |

### Palette Creation Approaches

**Monochromatic:** Single hue, vary saturation and lightness. Safe, sophisticated.

**Analogous:** Adjacent hues on color wheel. Harmonious, low contrast.

**Complementary:** Opposite hues. High contrast, energetic. Use sparingly.

**Triadic:** Three evenly spaced hues. Vibrant but harder to balance.

### Dark Mode Implementation

```css
/* Use CSS custom properties for theming */
:root {
  --bg: #ffffff;
  --surface: #f8fafc;
  --text: #0f172a;
  --text-secondary: #475569;
  --border: #e2e8f0;
}

[data-theme="dark"] {
  --bg: #0a0a0a;
  --surface: #171717;
  --text: #fafafa;
  --text-secondary: #a3a3a3;
  --border: #262626;
}

/* Respect system preference */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0a0a0a;
    --surface: #171717;
    --text: #fafafa;
    --text-secondary: #a3a3a3;
    --border: #262626;
  }
}
```

---

## Typography

### Type Scale

Use a consistent scale. Common ratios:
- **Minor Third** (1.2) — compact, minimal designs
- **Major Third** (1.25) — balanced, versatile
- **Perfect Fourth** (1.333) — generous, editorial
- **Golden Ratio** (1.618) — dramatic, luxury

### Font Pairing Rules

1. **Contrast** — pair serif display with sans-serif body (or vice versa)
2. **Max 2 families** — one display, one body. Add mono if needed for code.
3. **Match x-height** — fonts should feel balanced at the same size
4. **Match mood** — both fonts should share the same personality

### Recommended Pairings

| Display | Body | Mood |
|---------|------|------|
| Instrument Serif | Plus Jakarta Sans | Elegant modern |
| Space Grotesk | DM Sans | Tech/startup |
| Fraunces | Source Sans 3 | Editorial/literary |
| Sora | Inter Variable | Clean/minimal |
| Playfair Display | Lato | Luxury/premium |
| Libre Baskerville | Nunito Sans | Traditional/warm |
| Cabinet Grotesk | Satoshi | Bold/contemporary |

### Typography CSS

```css
/* Fluid typography with clamp */
h1 {
  font-family: var(--font-display);
  font-size: clamp(2rem, 5vw, 3.5rem);
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.02em;
}

h2 {
  font-family: var(--font-display);
  font-size: clamp(1.5rem, 3vw, 2.5rem);
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: -0.01em;
}

body {
  font-family: var(--font-body);
  font-size: clamp(1rem, 1.5vw, 1.125rem);
  line-height: 1.6;
  color: var(--text);
}

/* Paragraph spacing */
p + p { margin-top: 1.25em; }

/* Max line width for readability */
.prose { max-width: 65ch; }
```

---

## Accessibility (WCAG 2.1 AA)

### Contrast Requirements

| Element | Minimum Ratio |
|---------|--------------|
| Normal text (< 24px) | 4.5:1 |
| Large text (≥ 24px or 19px bold) | 3:1 |
| Interactive elements | 3:1 |
| Non-text (icons, borders) | 3:1 |

### Focus States

```css
/* Visible focus for keyboard users */
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}

/* Remove default for mouse users */
:focus:not(:focus-visible) {
  outline: none;
}

/* High-contrast focus for dark backgrounds */
.dark :focus-visible {
  outline-color: #ffffff;
  box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.2);
}
```

### Semantic HTML Checklist

- `<header>`, `<nav>`, `<main>`, `<footer>`, `<section>`, `<article>`
- `<button>` for actions, `<a>` for navigation
- `<h1>` through `<h6>` in order (no skipping levels)
- `alt` text on all `<img>` (empty `alt=""` for decorative)
- `<label>` associated with every form input
- `aria-label` for icon-only buttons
- `aria-live="polite"` for dynamic content updates
- `role="alert"` for error messages

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

---

## Responsive Patterns

### Breakpoint Strategy

```css
/* Mobile-first breakpoints */
/* sm: 640px  — Large phones, small tablets */
/* md: 768px  — Tablets */
/* lg: 1024px — Small laptops */
/* xl: 1280px — Desktops */
/* 2xl: 1536px — Large screens */

/* Container queries (modern approach) */
.card-grid {
  container-type: inline-size;
}

@container (min-width: 400px) {
  .card { flex-direction: row; }
}

@container (min-width: 700px) {
  .card-grid { grid-template-columns: repeat(2, 1fr); }
}
```

### Layout Patterns

```css
/* Fluid grid with auto-fill */
.grid-auto {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr));
  gap: var(--space-6);
}

/* Sidebar layout */
.with-sidebar {
  display: grid;
  grid-template-columns: minmax(250px, 25%) 1fr;
  gap: var(--space-8);
}

@media (max-width: 768px) {
  .with-sidebar {
    grid-template-columns: 1fr;
  }
}

/* Stack with consistent spacing */
.stack > * + * {
  margin-top: var(--space-4);
}

/* Center content with max-width */
.container {
  width: min(1200px, 100% - var(--space-8));
  margin-inline: auto;
}
```

---

## Animations and Transitions

### Entrance Animations

```css
/* Fade up on scroll (use Intersection Observer to trigger) */
.animate-fade-up {
  opacity: 0;
  transform: translateY(20px);
  transition: opacity var(--duration-normal) var(--ease-out),
              transform var(--duration-normal) var(--ease-out);
}

.animate-fade-up.visible {
  opacity: 1;
  transform: translateY(0);
}

/* Staggered children */
.stagger > * {
  opacity: 0;
  transform: translateY(10px);
  animation: fadeUp var(--duration-normal) var(--ease-out) forwards;
}

.stagger > *:nth-child(1) { animation-delay: 0ms; }
.stagger > *:nth-child(2) { animation-delay: 75ms; }
.stagger > *:nth-child(3) { animation-delay: 150ms; }
.stagger > *:nth-child(4) { animation-delay: 225ms; }

@keyframes fadeUp {
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### Micro-interactions

```css
/* Button hover with scale */
.btn {
  transition: transform var(--duration-fast) var(--ease-spring),
              box-shadow var(--duration-fast) var(--ease-out);
}

.btn:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.btn:active {
  transform: translateY(0) scale(0.98);
}

/* Card hover lift */
.card {
  transition: transform var(--duration-normal) var(--ease-out),
              box-shadow var(--duration-normal) var(--ease-out);
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-xl);
}

/* Skeleton loading */
.skeleton {
  background: linear-gradient(
    90deg,
    var(--surface) 25%,
    color-mix(in srgb, var(--surface), var(--text) 5%) 50%,
    var(--surface) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: var(--radius-md);
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

---

## Design Anti-Patterns (What NOT to Do)

### Visual Clichés to Avoid
- **Hero badges/pills** — "✨ Now with AI" badges above headings
- **Generic gradient CTAs** — purple-to-blue gradient buttons on white
- **Default Inter/Roboto** — use distinctive fonts that match your brand
- **Blob shapes** — abstract gradient blobs as decoration
- **Excessive rounded corners** — not everything needs `rounded-full`
- **Stock photo heroes** — use illustration, photography with character, or abstract

### Layout Mistakes
- **No visual hierarchy** — everything the same size and weight
- **Inconsistent spacing** — random padding/margin values
- **Text too wide** — body text exceeding 75 characters per line
- **No breathing room** — cramming content without whitespace
- **Centered everything** — center-aligned body text is hard to read

### Technical Mistakes
- **Layout shift** — content jumping as images/fonts load
- **Missing focus states** — keyboard users can't see where they are
- **Color-only indicators** — using only color to convey information
- **Tiny tap targets** — interactive elements smaller than 44×44px on mobile
- **Fixed font sizes** — using `px` instead of `rem` for text

---

## Design Review Checklist

- [ ] Clear visual hierarchy (headings, subheadings, body, captions)
- [ ] Consistent spacing using the design token scale
- [ ] All text passes contrast requirements (4.5:1 / 3:1)
- [ ] Visible focus states on all interactive elements
- [ ] Semantic HTML structure
- [ ] Mobile-responsive at all breakpoints
- [ ] Animations respect `prefers-reduced-motion`
- [ ] No generic/template aesthetic — intentional design choices
- [ ] Maximum line width ≤ 75ch for body text
- [ ] Tap targets ≥ 44×44px on mobile
- [ ] Alt text on images, labels on forms
- [ ] Consistent border radius, shadow, and color usage
- [ ] Loading states (skeleton, spinner) for async content

---

*Adapted from buildwithclaude by Dave Poon (MIT)*
