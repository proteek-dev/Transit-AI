// Transit AI web UI — inline SVG icons (bus, tram, train, location pin). Built with
// createElementNS from the shape specs below, so, like app.js, no markup
// string is ever parsed. Strokes use currentColor; CSS picks the colour.

(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  // 24x24 viewBox, 2px round strokes. Keyed by the API's leg.mode values.
  const SHAPES = {
    bus: [
      ['path', { d: 'M6 3h12a2 2 0 0 1 2 2v11a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a2 2 0 0 1 2-2z' }],
      ['path', { d: 'M4 11h16' }],
      ['path', { d: 'M8 7h8' }],
      ['circle', { cx: '7.5', cy: '19.5', r: '1.5' }],
      ['circle', { cx: '16.5', cy: '19.5', r: '1.5' }],
    ],
    tram: [
      ['path', { d: 'M8 2h8' }],
      ['path', { d: 'M12 2v4' }],
      ['path', { d: 'M7 6h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z' }],
      ['path', { d: 'M5 12h14' }],
      ['path', { d: 'M8 18l-2 4' }],
      ['path', { d: 'M16 18l2 4' }],
    ],
    rail: [
      ['path', { d: 'M8 2h8a3 3 0 0 1 3 3v10a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3V5a3 3 0 0 1 3-3z' }],
      ['path', { d: 'M5 10h14' }],
      ['circle', { cx: '9', cy: '14', r: '1' }],
      ['circle', { cx: '15', cy: '14', r: '1' }],
      ['path', { d: 'M8 18l-2 4' }],
      ['path', { d: 'M16 18l2 4' }],
    ],
  };

  // Location pin for the "Use my location" button.
  const PIN = [
    ['path', { d: 'M12 21s-6-5.4-6-10.5a6 6 0 0 1 12 0C18 15.6 12 21 12 21z' }],
    ['circle', { cx: '12', cy: '10.5', r: '2' }],
  ];

  // Decorative <svg> for `mode`, or null when there's no icon for it (the
  // text label beside it always carries the meaning).
  function modeIcon(mode, className) {
    const shapes = SHAPES[mode];
    return shapes ? build(shapes, className) : null;
  }

  function pinIcon(className) {
    return build(PIN, className);
  }

  function build(shapes, className) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    if (className) svg.setAttribute('class', className);
    for (const [tag, attrs] of shapes) {
      const node = document.createElementNS(SVG_NS, tag);
      for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
      svg.appendChild(node);
    }
    return svg;
  }

  window.TransitIcons = { modeIcon, pinIcon };
})();
