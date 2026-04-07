/**
 * Video Display Module
 * Renders video frames received via WebSocket onto an HTML5 canvas.
 */

class VideoDisplay {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('video-overlay');
        this.videoStatus = document.getElementById('video-status');
        this.frameCounter = document.getElementById('frame-counter');
        this.fpsCounter = document.getElementById('fps-counter');
        this.detectionCounter = document.getElementById('detection-counter');

        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fpsFrames = 0;
        this.currentFps = 0;
        this.isActive = false;

        // Offscreen image for decoding
        this._img = new Image();
        this._img.onload = () => {
            this._drawFrame();
        };
    }

    /**
     * Display a base64-encoded JPEG frame.
     * @param {string} base64Data - Base64 encoded JPEG image
     */
    displayFrame(base64Data) {
        this._img.src = 'data:image/jpeg;base64,' + base64Data;
        this.frameCount++;
        this.fpsFrames++;

        // Show video, hide overlay
        if (!this.isActive) {
            this.isActive = true;
            this.overlay.classList.add('hidden');
            this.videoStatus.textContent = 'LIVE';
            this.videoStatus.classList.add('active');
        }

        // Update frame counter
        if (this.frameCounter) {
            this.frameCounter.textContent = `Frame: ${this.frameCount}`;
        }

        // Calculate FPS every second
        const now = performance.now();
        const elapsed = now - this.lastFpsTime;
        if (elapsed >= 1000) {
            this.currentFps = Math.round((this.fpsFrames * 1000) / elapsed);
            this.fpsFrames = 0;
            this.lastFpsTime = now;

            if (this.fpsCounter) {
                this.fpsCounter.textContent = `${this.currentFps} FPS`;
            }
        }
    }

    /**
     * Update detection count display.
     */
    setDetectionCount(count) {
        if (this.detectionCounter) {
            this.detectionCounter.textContent = `${count} Detection${count !== 1 ? 's' : ''}`;
        }
    }

    /**
     * Draw the decoded image onto the canvas.
     */
    _drawFrame() {
        // Resize canvas to match image if needed
        if (this.canvas.width !== this._img.naturalWidth || 
            this.canvas.height !== this._img.naturalHeight) {
            this.canvas.width = this._img.naturalWidth;
            this.canvas.height = this._img.naturalHeight;
        }

        this.ctx.drawImage(this._img, 0, 0);
    }

    /**
     * Reset display to initial state.
     */
    reset() {
        this.isActive = false;
        this.frameCount = 0;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.overlay.classList.remove('hidden');
        this.videoStatus.textContent = 'OFFLINE';
        this.videoStatus.classList.remove('active');
    }
}

window.VideoDisplay = VideoDisplay;
