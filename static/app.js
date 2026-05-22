/**
 * app.js — Smart Reading Assistant Frontend Logic
 *
 * Vanilla JS — no frameworks. Handles tab switching, file upload with
 * drag-and-drop, RAG Q&A with confidence badges and source citations,
 * text summarization, document management, and toast notifications.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════════

const state = {
    documents: [],
    isUploading: false,
    isAsking: false,
    isAnalyzing: false,
};

// ═══════════════════════════════════════════════════════════════════════════════
// DOM HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initUpload();
    initChat();
    initSummarize();
    refreshDocumentList();
});

// ═══════════════════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════════════════

function initTabs() {
    $$(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;

            // Remove 'active' from all buttons and content panels
            $$(".tab-btn").forEach((b) => {
                b.classList.remove("active");
                b.setAttribute("aria-selected", "false");
            });
            $$(".tab-content").forEach((c) => c.classList.remove("active"));

            // Activate clicked button and matching tab panel
            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
            $(`#tab-${target}`).classList.add("active");
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOCUMENT UPLOAD
// ═══════════════════════════════════════════════════════════════════════════════

function initUpload() {
    const zone = $("#upload-zone");
    const input = $("#file-input");

    if (!zone || !input) return;

    // Clicking the upload zone triggers file input
    zone.addEventListener("click", (e) => {
        // Don't double-trigger if clicking the input itself
        if (e.target !== input) {
            input.click();
        }
    });

    // Drag-over: add accent border styling
    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("drag-over");
    });

    // Drag-leave: remove accent styling
    zone.addEventListener("dragleave", () => {
        zone.classList.remove("drag-over");
    });

    // Drop: handle dropped file
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // File input change handler
    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    if (state.isUploading) return;

    // Validate file extension
    const allowed = [".pdf", ".docx", ".txt"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
        showToast(`Unsupported file type "${ext}". Use PDF, DOCX, or TXT.`, "error");
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast("File too large. Maximum size is 10MB.", "error");
        return;
    }

    state.isUploading = true;

    // Step 1: Show progress
    const progress = $("#upload-progress");
    const barFill = $("#progress-bar-fill");
    const statusEl = $("#upload-status");

    if (progress) progress.classList.add("visible");
    if (statusEl) statusEl.textContent = `Uploading ${file.name}...`;

    // Step 2: Animate progress bar to 30% immediately
    if (barFill) {
        barFill.style.width = "0%";
        requestAnimationFrame(() => {
            barFill.style.width = "30%";
        });
    }

    // Fake progress animation
    let currentProgress = 30;
    const progressInterval = setInterval(() => {
        currentProgress += Math.random() * 10;
        if (currentProgress > 90) currentProgress = 90;
        if (barFill) barFill.style.width = currentProgress + "%";
    }, 300);

    const formData = new FormData();
    formData.append("file", file);

    try {
        // Step 3: POST to /upload
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        clearInterval(progressInterval);

        if (response.ok && data.status === "success") {
            // Step 4: Success — animate to 100%
            if (barFill) barFill.style.width = "100%";
            if (statusEl) statusEl.textContent = "Upload complete!";

            showToast(
                `✅ ${data.filename} — ${data.chunks_created} chunks created`,
                "success"
            );
            await refreshDocumentList();

            // Hide progress after 1.5s
            setTimeout(() => {
                if (progress) progress.classList.remove("visible");
                if (barFill) barFill.style.width = "0%";
            }, 1500);
        } else {
            // Step 5: Error
            showToast(`❌ ${data.error || data.message}`, "error");
            if (progress) progress.classList.remove("visible");
        }
    } catch (err) {
        clearInterval(progressInterval);
        showToast(`❌ Upload failed: ${err.message}`, "error");
        if (progress) progress.classList.remove("visible");
    } finally {
        state.isUploading = false;
        // Step 6: Always reset file input
        const input = $("#file-input");
        if (input) input.value = "";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOCUMENT MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

async function refreshDocumentList() {
    try {
        const response = await fetch("/documents");
        const data = await response.json();
        state.documents = data.documents || [];
        renderDocumentList();
    } catch (err) {
        console.error("Failed to load documents:", err);
    }
}

function renderDocumentList() {
    const list = $("#doc-list");
    const clearBtn = $("#btn-clear-docs");

    if (!list) return;

    if (state.documents.length === 0) {
        // Show empty placeholder, hide clear button
        list.innerHTML = '<div class="doc-empty">No documents uploaded yet</div>';
        if (clearBtn) clearBtn.style.display = "none";
        return;
    }

    // Show clear button
    if (clearBtn) clearBtn.style.display = "block";

    // Render each document
    list.innerHTML = state.documents
        .map(
            (doc) => `
        <div class="doc-item">
            <span class="doc-icon">📄</span>
            <span class="doc-filename" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
            <span class="doc-chunk-count">${doc.chunk_count} chunks</span>
        </div>
    `
        )
        .join("");
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLEAR DOCUMENTS
// ═══════════════════════════════════════════════════════════════════════════════

async function clearDocuments() {
    if (!confirm("Clear all documents? This cannot be undone.")) return;

    try {
        const response = await fetch("/documents", { method: "DELETE" });
        const data = await response.json();

        if (data.status === "success") {
            showToast("🗑️ All documents cleared", "info");
            state.documents = [];
            renderDocumentList();

            // Clear chat messages — restore welcome state
            const messages = $("#chat-messages");
            if (messages) {
                messages.innerHTML = `
                    <div class="welcome-state" id="welcome-state">
                        <span class="welcome-icon">💬</span>
                        <h3>Ask anything about your documents</h3>
                        <p>Upload a PDF, DOCX, or TXT file, then ask questions to get cited answers with confidence indicators.</p>
                    </div>
                `;
            }

            await refreshDocumentList();
        } else {
            showToast(`❌ ${data.message || "Failed to clear"}`, "error");
        }
    } catch (err) {
        showToast(`❌ Clear failed: ${err.message}`, "error");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASK QUESTION
// ═══════════════════════════════════════════════════════════════════════════════

function initChat() {
    const input = $("#chat-input");
    const sendBtn = $("#btn-send");

    if (!input || !sendBtn) return;

    // Click send button
    sendBtn.addEventListener("click", () => askQuestion());

    // Ctrl+Enter or Enter to send
    input.addEventListener("keydown", (e) => {
        if ((e.key === "Enter" && e.ctrlKey) || (e.key === "Enter" && !e.shiftKey)) {
            e.preventDefault();
            askQuestion();
        }
    });

    // Auto-resize textarea on input
    input.addEventListener("input", () => {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 120) + "px";
    });
}

async function askQuestion() {
    const input = $("#chat-input");
    const question = input.value.trim();

    // Step 1: Validate
    if (!question || state.isAsking) return;

    // Step 2: Clear input, disable send button
    input.value = "";
    input.style.height = "auto";
    state.isAsking = true;
    updateSendButton(true);

    // Step 3: Hide welcome state
    const welcome = $("#welcome-state");
    if (welcome) welcome.remove();

    // Step 4: Append question bubble
    const messages = $("#chat-messages");
    if (messages) {
        const questionDiv = document.createElement("div");
        questionDiv.className = "message message-question";
        questionDiv.textContent = question;
        messages.appendChild(questionDiv);
    }

    // Step 5: Append loading indicator
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message message-answer";
    loadingDiv.id = "loading-msg";
    loadingDiv.innerHTML = '<div class="loading-dots">Thinking...</div>';
    if (messages) messages.appendChild(loadingDiv);

    // Step 6: Scroll to bottom
    scrollChatToBottom();

    try {
        // Step 7: POST to /ask
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();

        // Step 8: Remove loading indicator
        const loading = document.getElementById("loading-msg");
        if (loading) loading.remove();

        if (response.ok) {
            // Step 9: Render answer
            renderAnswer(data);
        } else {
            // Step 10: Error bubble
            addErrorMessage(`Error: ${data.error || "Failed to get answer"}`);
        }
    } catch (err) {
        const loading = document.getElementById("loading-msg");
        if (loading) loading.remove();
        addErrorMessage(`Error: ${err.message}`);
    } finally {
        // Step 11: Re-enable send button
        state.isAsking = false;
        updateSendButton(false);
    }
}

function renderAnswer(result) {
    const messages = $("#chat-messages");
    if (!messages) return;

    const div = document.createElement("div");
    div.className = "message message-answer";

    // Confidence badge
    const confidenceClass = result.confidence ? result.confidence.toLowerCase() : "not_found";
    const confidenceIcons = { HIGH: "✅", PARTIAL: "⚠️", NOT_FOUND: "❌" };
    const confidenceIcon = confidenceIcons[result.confidence] || "❌";

    // Answer text
    const answerHtml = formatAnswer(result.answer || "No answer available.");

    // Partial note
    const partialNoteHtml = result.partial_note
        ? `<div class="partial-note">ℹ️ ${escapeHtml(result.partial_note)}</div>`
        : "";

    // Sources
    let sourcesHtml = "";
    if (result.sources && result.sources.length > 0) {
        const sourceId = "sources-" + Date.now();
        const sourceItems = result.sources
            .map(
                (s) => `
            <div class="source-item">
                <div class="source-filename">📄 ${escapeHtml(s.filename)}</div>
                <div class="source-preview">${escapeHtml(s.chunk_preview)}</div>
                <span class="source-page">Page ${s.page + 1}</span>
            </div>
        `
            )
            .join("");

        sourcesHtml = `
            <button class="sources-toggle" onclick="toggleSources(this)">
                📎 ${result.sources.length} source(s) — click to expand
            </button>
            <div class="sources-panel">
                ${sourceItems}
            </div>
        `;
    }

    div.innerHTML = `
        <span class="confidence-badge confidence-${confidenceClass}">
            ${confidenceIcon} ${result.confidence || "UNKNOWN"}
        </span>
        <div class="answer-text">${answerHtml}</div>
        ${partialNoteHtml}
        ${sourcesHtml}
    `;

    messages.appendChild(div);
    scrollChatToBottom();
}

function toggleSources(btn) {
    const panel = btn.nextElementSibling;
    if (!panel) return;

    const isOpen = panel.classList.toggle("open");
    btn.textContent = isOpen
        ? `📎 Sources — click to collapse`
        : `📎 Sources — click to expand`;
}

function addErrorMessage(text) {
    const messages = $("#chat-messages");
    if (!messages) return;

    const div = document.createElement("div");
    div.className = "message message-error";
    div.textContent = text;
    messages.appendChild(div);
    scrollChatToBottom();
}

function updateSendButton(loading) {
    const btn = $("#btn-send");
    if (!btn) return;

    btn.disabled = loading;
    btn.innerHTML = loading
        ? '<div class="spinner"></div>'
        : "🔍 Ask";
}

function scrollChatToBottom() {
    const messages = $("#chat-messages");
    if (messages) {
        messages.scrollTop = messages.scrollHeight;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUMMARIZE TAB
// ═══════════════════════════════════════════════════════════════════════════════

function initSummarize() {
    const btn = $("#btn-analyze");
    if (!btn) return;

    btn.addEventListener("click", () => runAnalysis());
}

async function runAnalysis() {
    const textarea = $("#summarize-input");
    const text = textarea ? textarea.value.trim() : "";

    // Step 1: Validate
    if (!text) {
        showToast("Please paste some text to analyze.", "error");
        return;
    }

    if (state.isAnalyzing) return;

    // Step 2: Set button to loading state
    state.isAnalyzing = true;
    const btn = $("#btn-analyze");
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Analyzing...';
    }

    // Step 3: POST to /analyze
    const formData = new FormData();
    formData.append("text", text);

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            // Step 4: Show results
            const section = $("#results-section");
            const summaryEl = $("#result-summary");
            const phrasesEl = $("#result-phrases");

            if (section) section.classList.add("visible");

            if (summaryEl) {
                summaryEl.textContent = data.summary || "No summary generated.";
            }

            if (phrasesEl) {
                const phrases = data.key_phrases || [];
                phrasesEl.innerHTML = phrases
                    .map((p) => `<li class="key-phrase-item">${escapeHtml(p)}</li>`)
                    .join("");
            }
        } else {
            // Step 5: Error toast
            showToast(`❌ ${data.error || "Analysis failed"}`, "error");
        }
    } catch (err) {
        showToast(`❌ Analysis failed: ${err.message}`, "error");
    } finally {
        // Step 6: Re-enable button
        state.isAnalyzing = false;
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = "✨ Analyze Text";
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOAST SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

function showToast(message, type = "info", duration = 4000) {
    let container = $("#toast-container");
    if (!container) {
        container = document.createElement("div");
        container.className = "toast-container";
        container.id = "toast-container";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    // Auto-remove after duration
    setTimeout(() => {
        toast.style.animation = "toastOut 0.3s ease forwards";
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    // Basic markdown-like formatting for LLM responses
    return escapeHtml(text)
        .replace(/\n\n/g, "</p><p>")
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>");
}
