const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 }
  });
  const page = await context.newPage();

  try {
    await page.goto('http://127.0.0.1:18080', { waitUntil: 'networkidle', timeout: 30000 });
    // Wait 3 seconds after initial load for chart to render
    await page.waitForTimeout(3000);

    const screenshotPath = 'g:\\GitHub\\Latent_Style\\SchrodingerBridge\\chart_screenshot.png';
    await page.screenshot({ path: screenshotPath, fullPage: false });
    console.log('Screenshot saved to:', screenshotPath);
  } catch (err) {
    console.error('Error:', err.message);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
