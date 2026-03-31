// <fe-doc-viewer> — Composable documentation viewer for Fe.
//
// Composes <fe-doc-nav> + content rendering + routing into a single
// drop-in component.  The host page can use this for a full-featured
// doc browser, or use the sub-components individually for custom layouts.
//
// Usage:
//   <fe-doc-viewer src="/docs.json" title="Fe Std Library"
//     back-href="/" back-label="Back to Guide" />
//
// Attributes:
//   src          — URL to docs.json (required)
//   title        — header title text
//   back-href    — URL for the back/home link
//   back-label   — text for the back/home link
//   routing      — "hash" (default), "path", or "none"
//   base         — base URL for path-based routing
//   filter       — passed to <fe-doc-nav> and content
//   filter-kind  — passed to <fe-doc-nav>
//   exclude      — passed to <fe-doc-nav> and content

class FeDocViewer extends HTMLElement {
  static get observedAttributes() {
    return ["src", "title", "routing", "filter", "filter-kind", "exclude"];
  }

  attributeChangedCallback(name, oldVal, newVal) {
    if (oldVal === newVal) return;
    if (name === "src") { this._loadSrc(); return; }
    this._render();
  }

  connectedCallback() {
    this._createStructure();
    this._loadSrc();
    this._setupRouting();
  }

  disconnectedCallback() {
    if (this._hashHandler) {
      window.removeEventListener("hashchange", this._hashHandler);
      this._hashHandler = null;
    }
  }

  _createStructure() {
    this.innerHTML = "";
    this.classList.add("fe-doc-viewer");

    // Header
    var header = document.createElement("div");
    header.className = "fe-doc-viewer-header";
    this._headerEl = header;
    this.appendChild(header);

    // Layout container
    var layout = document.createElement("div");
    layout.className = "fe-doc-viewer-layout";

    // Nav
    var nav = document.createElement("fe-doc-nav");
    nav.setAttribute("show-search", "");
    this._navEl = nav;
    layout.appendChild(nav);

    // Content
    var content = document.createElement("div");
    content.className = "fe-doc-viewer-content";
    this._contentEl = content;
    layout.appendChild(content);

    this.appendChild(layout);

    // Listen for navigation events from nav and content
    var self = this;
    this.addEventListener("fe-navigate", function (e) {
      var docPath = e.detail.docPath;
      if (!docPath) return;

      var routing = self.getAttribute("routing") || "hash";
      if (routing === "hash") {
        location.hash = "#" + docPath;
      } else if (routing === "path") {
        var base = self.getAttribute("base") || "/";
        history.pushState(null, "", base + docPath);
        self._showItem(docPath);
      } else {
        // routing="none" — just render in place
        self._showItem(docPath);
      }
      e.stopPropagation();
    });
  }

  _loadSrc() {
    var src = this.getAttribute("src");
    if (!src) return;
    var self = this;

    // Pass src to nav
    if (this._navEl) {
      this._navEl.setAttribute("src", src);
    }

    feLoadSrc(src).then(function (result) {
      self._index = result.index;
      self._scip = result.scip;
      self._render();

      // Show initial item from hash
      var routing = self.getAttribute("routing") || "hash";
      if (routing === "hash") {
        var path = location.hash.replace(/^#\/?/, "");
        if (path) self._showItem(decodeURIComponent(path));
        else self._showWelcome();
      } else {
        self._showWelcome();
      }
    });
  }

  _setupRouting() {
    var routing = this.getAttribute("routing") || "hash";
    if (routing !== "hash") return;

    var self = this;
    this._hashHandler = function () {
      var path = location.hash.replace(/^#\/?/, "");
      if (path) {
        self._showItem(decodeURIComponent(path));
        if (self._navEl) self._navEl.setAttribute("active", decodeURIComponent(path));
      } else {
        self._showWelcome();
      }
    };
    window.addEventListener("hashchange", this._hashHandler);
  }

  _render() {
    // Update header
    if (this._headerEl) {
      var html = "";
      var backHref = this.getAttribute("back-href");
      var backLabel = this.getAttribute("back-label");
      if (backHref) {
        html += '<a class="fe-doc-viewer-back" href="' + _feViewerEsc(backHref) + '">' +
          _feViewerEsc(backLabel || "Back") + "</a>";
      }
      var title = this.getAttribute("title");
      if (title) {
        html += '<span class="fe-doc-viewer-title">' + _feViewerEsc(title) + "</span>";
      }
      this._headerEl.innerHTML = html;
    }

    // Pass filter attributes to nav
    if (this._navEl) {
      var attrs = ["filter", "filter-kind", "exclude"];
      for (var i = 0; i < attrs.length; i++) {
        var val = this.getAttribute(attrs[i]);
        if (val) this._navEl.setAttribute(attrs[i], val);
        else this._navEl.removeAttribute(attrs[i]);
      }
    }
  }

  _showItem(docPath) {
    if (!this._contentEl) return;
    var index = this._index || window.FE_DOC_INDEX;
    if (!index || !index.items) return;

    // Update nav active state
    if (this._navEl) this._navEl.setAttribute("active", docPath);

    // Find item by URL path
    var item = this._findByUrl(index, docPath);
    if (!item) {
      this._contentEl.innerHTML = '<div class="fe-doc-viewer-not-found">' +
        '<p>Item not found: <code>' + _feViewerEsc(docPath) + '</code></p></div>';
      return;
    }

    // Render via <fe-doc-item>
    var docItem = document.createElement("fe-doc-item");
    docItem.setAttribute("symbol", item.path);
    if (this.getAttribute("src")) docItem.setAttribute("src", this.getAttribute("src"));
    if (this.getAttribute("base")) docItem.setAttribute("base", this.getAttribute("base"));

    // Pass through filter attributes
    var filterAttrs = ["filter", "filter-kind", "exclude"];
    for (var i = 0; i < filterAttrs.length; i++) {
      var val = this.getAttribute(filterAttrs[i]);
      if (val) docItem.setAttribute(filterAttrs[i], val);
    }

    this._contentEl.innerHTML = "";
    this._contentEl.appendChild(docItem);
  }

  _showWelcome() {
    if (!this._contentEl) return;
    var index = this._index || window.FE_DOC_INDEX;
    if (!index) {
      this._contentEl.innerHTML = '<div class="fe-doc-viewer-welcome">' +
        '<p>Loading documentation...</p></div>';
      return;
    }

    var title = this.getAttribute("title") || "Fe Documentation";
    var base = this.getAttribute("base") || "";
    var modules = index.modules || [];
    var builtinModules = index.builtin_modules || [];
    var items = index.items || [];

    // Find root module and its DocItem
    var rootMod = modules[0] || builtinModules[0] || null;
    var rootDocItem = null;
    if (rootMod) {
      for (var i = 0; i < items.length; i++) {
        if (items[i].path === rootMod.path && items[i].kind === "module") {
          rootDocItem = items[i]; break;
        }
      }
    }

    // If we found a root module, show it as the landing page (like docs.rs)
    if (rootMod) {
      var modUrl = rootMod.path + "/mod";
      this._showItem(modUrl);
      if (this._navEl) this._navEl.setAttribute("active", modUrl);
      return;
    }

    // Fallback: no modules at all
    this._contentEl.innerHTML = '<div class="fe-doc-viewer-welcome">' +
      '<h1>' + _feViewerEsc(title) + '</h1>' +
      '<p>No documented items found.</p></div>';
  }

  _findByUrl(index, urlPath) {
    if (!urlPath) return null;
    var items = index.items || [];

    var slashIdx = urlPath.lastIndexOf("/");
    if (slashIdx !== -1) {
      var path = urlPath.substring(0, slashIdx);
      var kindSuffix = urlPath.substring(slashIdx + 1);
      var kindMap = {
        mod: "module", fn: "function", struct: "struct", enum: "enum",
        trait: "trait", contract: "contract", type: "type_alias",
        "const": "const", impl: "impl",
      };
      var kindName = kindMap[kindSuffix];
      if (kindName) {
        for (var i = 0; i < items.length; i++) {
          if (items[i].path === path && items[i].kind === kindName) return items[i];
        }
      }
    }

    for (var j = 0; j < items.length; j++) {
      if (items[j].path === urlPath) return items[j];
    }
    return null;
  }
}

function _feViewerEsc(s) {
  var d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

customElements.define("fe-doc-viewer", FeDocViewer);
