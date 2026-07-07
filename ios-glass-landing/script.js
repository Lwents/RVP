/* ============================================
   iOS 26 Liquid Glass — JavaScript
   ============================================ */

(function () {
  'use strict';

  // ---------- Navbar scroll effect ----------
  const navbar = document.getElementById('navbar');

  function updateNavbar() {
    if (!navbar) return;
    if (window.scrollY > 12) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }

  window.addEventListener('scroll', updateNavbar, { passive: true });
  updateNavbar();

  // ---------- Mobile menu toggle ----------
  const navToggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');
  const menuOverlay = document.getElementById('menu-overlay');

  function openMenu() {
    navToggle.classList.add('active');
    navLinks.classList.add('open');
    menuOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeMenu() {
    navToggle.classList.remove('active');
    navLinks.classList.remove('open');
    menuOverlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (navToggle) {
    navToggle.addEventListener('click', function () {
      const isOpen = navLinks.classList.contains('open');
      isOpen ? closeMenu() : openMenu();
    });
  }

  if (menuOverlay) {
    menuOverlay.addEventListener('click', closeMenu);
  }

  // Close menu when a link is clicked
  if (navLinks) {
    navLinks.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', closeMenu);
    });
  }

  // Auto-close mobile menu on viewport resize past breakpoint
  var mobileBreakpoint = 680;
  window.addEventListener('resize', function () {
    if (window.innerWidth > mobileBreakpoint) {
      closeMenu();
    }
  });

  // ---------- Scroll reveal ----------
  const revealElements = document.querySelectorAll('.reveal');

  const revealObserver = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.12,
      rootMargin: '0px 0px -40px 0px',
    }
  );

  revealElements.forEach(function (el) {
    revealObserver.observe(el);
  });

  // ---------- Stats counter animation ----------
  const statValues = document.querySelectorAll('.stat-item__value[data-count]');

  const counterObserver = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.5 }
  );

  statValues.forEach(function (el) {
    counterObserver.observe(el);
  });

  function animateCounter(el) {
    var target = parseInt(el.getAttribute('data-count'), 10);
    var suffix = '';

    // Special case for the "0KB" stat
    if (target === 0 && el.textContent.includes('KB')) {
      el.textContent = '0KB';
      return;
    }

    var duration = 1400;
    var startTime = null;

    function step(timestamp) {
      if (!startTime) startTime = timestamp;
      var progress = Math.min((timestamp - startTime) / duration, 1);
      // Ease out cubic
      var easedProgress = 1 - Math.pow(1 - progress, 3);
      var current = Math.round(easedProgress * target);
      el.textContent = current + suffix;
      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  // ---------- Smooth scroll for anchor links ----------
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      var targetId = this.getAttribute('href');
      if (targetId === '#') return;
      var targetEl = document.querySelector(targetId);
      if (targetEl) {
        e.preventDefault();
        var navHeight = navbar ? navbar.offsetHeight : 0;
        var top = targetEl.getBoundingClientRect().top + window.pageYOffset - navHeight - 16;
        window.scrollTo({ top: top, behavior: 'smooth' });
      }
    });
  });

  // ---------- CTA form feedback ----------
  var ctaForm = document.getElementById('cta-form');
  var ctaSubmit = document.getElementById('cta-submit');
  var ctaEmail = document.getElementById('cta-email');

  if (ctaForm) {
    ctaForm.addEventListener('submit', function (e) {
      e.preventDefault();
      if (!ctaEmail.value || !ctaEmail.validity.valid) {
        ctaEmail.focus();
        return;
      }

      // Visual feedback
      var originalText = ctaSubmit.textContent;
      ctaSubmit.textContent = '✓ Đã gửi!';
      ctaSubmit.style.background = '#34c759';
      ctaSubmit.style.boxShadow = '0 10px 30px rgba(52, 199, 89, 0.35)';
      ctaSubmit.disabled = true;
      ctaEmail.value = '';

      setTimeout(function () {
        ctaSubmit.textContent = originalText;
        ctaSubmit.style.background = '';
        ctaSubmit.style.boxShadow = '';
        ctaSubmit.disabled = false;
      }, 2500);
    });
  }

  // ---------- Keyboard accessibility ----------
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
      closeMenu();
    }
  });

})();
