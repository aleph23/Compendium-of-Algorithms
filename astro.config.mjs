// astro.config.mjs
import starlight from '@astrojs/starlight'
import { defineConfig } from 'astro/config'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

const BASE = process.env.ASTRO_BASE || "/";

export default defineConfig({
  base: BASE,
  outDir: "src/content/docs",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: "Compendium of Algorithms",
      description:
        "The Formal Specification and Structural Composition of Probabilistic " +
        "Computational Systems and Their Occasional Interdependence",
      head: [
        { tag: "link", attrs: { rel: "stylesheet", href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" } },
        { tag: "link", attrs: { rel: "preconnect", href: "https://fonts.googleapis.com" } },
        { tag: "link", attrs: { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" } },
      ],
      components: { Head: "./src/components/CompendiumHead.astro", },
        social: { github: "https://github.com/aleph23/compendium-of-algorithms", },
      sidebar: [ 
        { 
          label: "Liber Primus: Received Canon",
          items: [
            { label: "Introduction", link: "/received-canon/" },
            { label: "Foundational", autogenerate: { directory: "/received-canon/foundational" } },
            { label: "Convolutional Networks", autogenerate: { directory: "/received-canon/convolutional" } },
            { label: "Recurrent Networks", autogenerate: { directory: "/received-canon/recurrent" } },
            { label: "Attention & Transformers", autogenerate: { directory: "/received-canon/attention" } },
            { label: "Generative Models", autogenerate: { directory: "/received-canon/generative" } },
            { label: "Graph & Geometric", autogenerate: { directory: "/received-canon/graph" } },
            { label: "Reinforcement Learning", autogenerate: { directory: "/received-canon/reinforcement" } },
            { label: "Modern Architectures", autogenerate: { directory: "/received-canon/modern" } },
          ],
        },
        {
          label: "Liber Levitatis et Inconstantiae: Frontier",
          items: [
            { label: "About Frontier", link: "/frontier/" },
            { label: "Emergent Architectures", autogenerate: { directory: "/frontier/emergent" } },
            { label: "Techniques Under Review", autogenerate: { directory: "/frontier/under-review" } },
            { label: "Archived Editions", autogenerate: { directory: "/frontier/archive" } },
          ],
        },
      ],
      customCss: [
        "./src/styles/received-canon.css",
        "./src/styles/frontier.css",
        "./src/styles/custom.css",
      ],
      lastUpdated: true,
      pagination: true,
    }),
  ],
});
