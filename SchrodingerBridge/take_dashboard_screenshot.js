const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1456, height: 819 }
  });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:8080', { waitUntil: 'networkidle' });
  // Wait an additional 5 seconds for charts/dashboard to render
  await page.waitForTimeout(5000);
  await page.screenshot({
    path: 'G:\\GitHub\\Latent_Style\\SchrodingerBridge\\actual_dashboard_v3.png',
    fullPage: false
  });
  await browser.close();
  console.log('Screenshot saved to actual_dashboard_v3.png');
})();
