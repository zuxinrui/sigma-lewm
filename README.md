# Σ-LeWM project page

Static site, no build step. `index.html` + `fig/` is everything the page needs
(the title animation is rendered live by the inline script; `title.mp4` is the
same animation as a 1080p60 video, for sharing or as a fallback).

This is the `gh-pages` branch of https://github.com/zuxinrui/sigma-lewm; the code
lives on `main`. Once the repository is public, serve it at
https://zuxinrui.github.io/sigma-lewm/ with:

    gh api -X POST repos/zuxinrui/sigma-lewm/pages -f 'source[branch]=gh-pages' -f 'source[path]=/'

(`.nojekyll` tells Pages to serve the files as-is.)
