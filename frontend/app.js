/**
 * GreenLeaf Grocery - Demand Forecasting Dashboard
 * Dynamic JavaScript Controller
 * Handles SKU auto-filling, API interaction, validation, and inventory recommendations.
 */

// 1. Full 12-Item Organic Produce Catalogue (Matching ML Model Training Data)
const PRODUCE_CATALOGUE = {
  SKU_001: {
    id: "SKU_001",
    name: "Organic Bananas",
    category: "Fruit",
    defaultPrice: 1.99,
    defaultStock: 85,
    icon: "🍌"
  },
  SKU_002: {
    id: "SKU_002",
    name: "Baby Spinach",
    category: "Leafy Greens",
    defaultPrice: 3.49,
    defaultStock: 45,
    icon: "🥬"
  },
  SKU_003: {
    id: "SKU_003",
    name: "Roma Tomatoes",
    category: "Vegetables",
    defaultPrice: 2.29,
    defaultStock: 60,
    icon: "🍅"
  },
  SKU_004: {
    id: "SKU_004",
    name: "Hass Avocados",
    category: "Fruit",
    defaultPrice: 4.99,
    defaultStock: 50,
    icon: "🥑"
  },
  SKU_005: {
    id: "SKU_005",
    name: "Organic Gala Apples",
    category: "Fruit",
    defaultPrice: 3.99,
    defaultStock: 75,
    icon: "🍎"
  },
  SKU_006: {
    id: "SKU_006",
    name: "Fresh Strawberries",
    category: "Berries",
    defaultPrice: 4.49,
    defaultStock: 40,
    icon: "🍓"
  },
  SKU_007: {
    id: "SKU_007",
    name: "Organic Broccoli",
    category: "Vegetables",
    defaultPrice: 2.89,
    defaultStock: 35,
    icon: "🥦"
  },
  SKU_008: {
    id: "SKU_008",
    name: "Red Bell Peppers",
    category: "Vegetables",
    defaultPrice: 3.19,
    defaultStock: 30,
    icon: "🫑"
  },
  SKU_009: {
    id: "SKU_009",
    name: "English Cucumbers",
    category: "Vegetables",
    defaultPrice: 1.79,
    defaultStock: 40,
    icon: "🥒"
  },
  SKU_010: {
    id: "SKU_010",
    name: "Organic Carrots",
    category: "Vegetables",
    defaultPrice: 2.19,
    defaultStock: 50,
    icon: "🥕"
  },
  SKU_011: {
    id: "SKU_011",
    name: "Fresh Blueberries",
    category: "Berries",
    defaultPrice: 4.99,
    defaultStock: 30,
    icon: "🫐"
  },
  SKU_012: {
    id: "SKU_012",
    name: "Organic Tuscan Kale",
    category: "Leafy Greens",
    defaultPrice: 2.99,
    defaultStock: 25,
    icon: "🥗"
  }
};

// 2. DOM Elements
const form = document.getElementById("forecast-form");
const skuSelect = document.getElementById("sku_id");
const productNameInput = document.getElementById("product_name");
const categorySelect = document.getElementById("category");
const unitPriceInput = document.getElementById("unit_price");
const inventorySlider = document.getElementById("inventory_level");
const inventoryValBadge = document.getElementById("inventory_val");
const promotionToggle = document.getElementById("promotion_toggle");
const dateInput = document.getElementById("forecast_date");
const submitBtn = document.getElementById("submit_btn");
const submitSpinner = document.getElementById("btn_spinner");
const submitIcon = document.getElementById("btn_icon");
const submitText = document.getElementById("btn_text");
const errorBanner = document.getElementById("error_banner");

// Output Elements
const resultPlaceholder = document.getElementById("result_placeholder");
const resultActive = document.getElementById("result_active");
const predictedValueEl = document.getElementById("predicted_value");
const recTitleEl = document.getElementById("rec_title");
const recMessageEl = document.getElementById("rec_message");
const progressFillEl = document.getElementById("progress_fill");
const metaSkuEl = document.getElementById("meta_sku");
const metaCategoryEl = document.getElementById("meta_category");
const metaPriceEl = document.getElementById("meta_price");
const metaStockEl = document.getElementById("meta_stock");

// 3. Initialize Dashboard Defaults
function initializeDashboard() {
  // Set default date to tomorrow
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  dateInput.value = tomorrow.toISOString().split("T")[0];

  // Populate SKU dropdown
  skuSelect.innerHTML = "";
  Object.values(PRODUCE_CATALOGUE).forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = `${item.icon} ${item.name} (${item.id})`;
    skuSelect.appendChild(opt);
  });

  // Select initial SKU
  selectProduceSKU("SKU_001");

  // Event Listeners
  inventorySlider.addEventListener("input", (e) => {
    inventoryValBadge.textContent = e.target.value;
  });

  skuSelect.addEventListener("change", (e) => {
    selectProduceSKU(e.target.value);
  });

  // Preset chip button clicks
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      const skuId = chip.getAttribute("data-sku");
      skuSelect.value = skuId;
      selectProduceSKU(skuId);
    });
  });

  // Form submit handler
  form.addEventListener("submit", handleForecastSubmit);
}

// 4. SKU Selection & Auto-Fill
function selectProduceSKU(skuId) {
  const item = PRODUCE_CATALOGUE[skuId];
  if (!item) return;

  productNameInput.value = item.name;
  categorySelect.value = item.category;
  unitPriceInput.value = item.defaultPrice.toFixed(2);
  inventorySlider.value = item.defaultStock;
  inventoryValBadge.textContent = item.defaultStock;

  // Highlight active preset chip
  document.querySelectorAll(".preset-chip").forEach((c) => {
    if (c.getAttribute("data-sku") === skuId) {
      c.classList.add("active");
    } else {
      c.classList.remove("active");
    }
  });
}

// 5. Handle Forecast Submission & API Call
async function handleForecastSubmit(e) {
  e.preventDefault();
  hideError();
  setLoading(true);

  // Build Payload conforming to Pydantic PredictionRequest schema
  const payload = {
    date: dateInput.value,
    sku_id: skuSelect.value,
    product_name: productNameInput.value,
    category: categorySelect.value,
    unit_price: parseFloat(unitPriceInput.value),
    inventory_level: parseInt(inventorySlider.value, 10),
    promotion: promotionToggle.checked ? "Yes" : "No"
  };

  // Determine API base URL (relative if hosted on same origin, or fallback)
  const apiEndpoint = "/predict";

  try {
    const response = await fetch(apiEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      let errMsg = `Request failed with status ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson.detail) {
          if (Array.isArray(errJson.detail)) {
            errMsg = errJson.detail.map((d) => d.msg || JSON.stringify(d)).join(", ");
          } else {
            errMsg = errJson.detail;
          }
        }
      } catch (_) {}
      throw new Error(errMsg);
    }

    const data = await response.json();
    renderForecastResults(data, payload);
  } catch (err) {
    showError(err.message || "Failed to communicate with prediction service.");
  } finally {
    setLoading(false);
  }
}

// 6. Render Prediction Output & Business Recommendations
function renderForecastResults(predictionData, requestPayload) {
  const predictedUnits = predictionData.predicted_demand;
  const currentStock = requestPayload.inventory_level;

  // Update hero forecast display
  predictedValueEl.textContent = predictedUnits.toFixed(1);

  // Calculate stock balance & replenishment advice
  const stockDifference = currentStock - predictedUnits;
  let statusColor = "#2d6a4f";
  let titleText = "Optimal Inventory Level";
  let messageText = "";

  if (stockDifference < 0) {
    // Understocked (Demand exceeds stock)
    const shortage = Math.abs(stockDifference);
    const safetyBuffer = Math.ceil(shortage * 0.15); // 15% perishable buffer
    const recommendedOrder = Math.ceil(shortage + safetyBuffer);
    statusColor = "#e76f51"; // Alert red-orange
    titleText = `⚠️ Understock Risk (${shortage.toFixed(0)} units deficit)`;
    messageText = `Expected demand exceeds current stock. Recommended re-order: <strong>${recommendedOrder} units</strong> (includes 15% perishable freshness safety buffer).`;
  } else if (stockDifference > predictedUnits * 0.5) {
    // Overstock risk (perishable waste hazard)
    statusColor = "#d97706"; // Amber
    titleText = "⚠️ High Inventory Alert (Perishable Spoilage Risk)";
    messageText = `Current stock (${currentStock} units) substantially exceeds tomorrow's forecasted demand. Consider launching a store promotion or reducing next order.`;
  } else {
    // Balanced
    statusColor = "#2d6a4f"; // Green
    titleText = "✅ Well-Balanced Inventory";
    messageText = `Current stock (${currentStock} units) comfortably covers forecasted demand (${predictedUnits.toFixed(1)} units) with minimal spoilage risk.`;
  }

  recTitleEl.innerHTML = titleText;
  recTitleEl.style.color = statusColor;
  recMessageEl.innerHTML = messageText;

  // Progress Bar (Stock vs Demand coverage)
  const coveragePercent = Math.min(100, Math.round((currentStock / (predictedUnits || 1)) * 100));
  progressFillEl.style.width = `${coveragePercent}%`;
  progressFillEl.style.backgroundColor = statusColor;

  // Meta statistics table
  metaSkuEl.textContent = requestPayload.sku_id;
  metaCategoryEl.textContent = requestPayload.category;
  metaPriceEl.textContent = `$${requestPayload.unit_price.toFixed(2)}`;
  metaStockEl.textContent = `${currentStock} units`;

  // Display result panel
  resultPlaceholder.style.display = "none";
  resultActive.style.display = "block";
}

// 7. Utility States
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitSpinner.style.display = isLoading ? "inline-block" : "none";
  submitIcon.style.display = isLoading ? "none" : "inline-block";
  submitText.textContent = isLoading ? "Calculating Forecast..." : "Predict Daily Demand";
}

function showError(msg) {
  errorBanner.textContent = `Error: ${msg}`;
  errorBanner.style.display = "block";
}

function hideError() {
  errorBanner.style.display = "none";
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", initializeDashboard);
