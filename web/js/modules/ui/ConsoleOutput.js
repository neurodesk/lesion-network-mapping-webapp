/**
 * Console Output Module
 * Handles logging output to the UI console panel.
 */
export class ConsoleOutput {
  constructor(outputElementId = 'consoleOutput') {
    this.outputElementId = outputElementId;
  }

  log(text) {
    const outputElement = document.getElementById(this.outputElementId);
    if (outputElement) {
      const time = new Date().toLocaleTimeString('en-US', { hour12: false });
      const line = document.createElement('div');
      line.className = 'console-line';
      line.innerHTML = `<span class="console-time">[${time}]</span> <span class="console-message">${text}</span>`;
      outputElement.appendChild(line);
      outputElement.scrollTop = outputElement.scrollHeight;
    }
    console.log(text);
  }

  async copyToClipboard() {
    const outputElement = document.getElementById(this.outputElementId);
    if (!outputElement) return false;
    const lines = outputElement.querySelectorAll('.console-line');
    const text = Array.from(lines).map(line => line.textContent).join('\n');
    try {
      await this.writeTextToClipboard(text);
      this.setCopyButtonText('Copied!');
      return true;
    } catch (err) {
      this.setCopyButtonText('Copy failed');
      console.warn('Console copy failed:', err);
      return false;
    }
  }

  async writeTextToClipboard(text) {
    const clipboard = globalThis.navigator?.clipboard;
    if (clipboard?.writeText) {
      try {
        await clipboard.writeText(text);
        return;
      } catch (err) {
        // Fall back below for in-app browsers or permission-denied contexts.
      }
    }

    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    let copied = false;
    try {
      copied = !!document.execCommand?.('copy');
    } finally {
      textarea.remove();
    }
    if (!copied) throw new Error('document.execCommand("copy") failed');
  }

  setCopyButtonText(text) {
    const btn = document.getElementById('copyConsole');
    if (!btn) return;
    btn.textContent = text;
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  }

  clear() {
    const outputElement = document.getElementById(this.outputElementId);
    if (outputElement) {
      outputElement.innerHTML = '';
    }
  }
}
