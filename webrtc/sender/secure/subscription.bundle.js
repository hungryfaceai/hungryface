var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __esm = (fn, res, err) => function __init() {
  if (err) throw err[0];
  try {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  } catch (e) {
    throw err = [e], e;
  }
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// node_modules/@capacitor/core/dist/index.js
var ExceptionCode, CapacitorException, getPlatformId, createCapacitor, initCapacitorGlobal, Capacitor, registerPlugin, WebPlugin, encode, decode, CapacitorCookiesPluginWeb, CapacitorCookies, readBlobAsBase64, normalizeHttpHeaders, buildUrlParams, buildRequestInit, CapacitorHttpPluginWeb, CapacitorHttp, SystemBarsStyle, SystemBarType, SystemBarsPluginWeb, SystemBars;
var init_dist = __esm({
  "node_modules/@capacitor/core/dist/index.js"() {
    (function(ExceptionCode2) {
      ExceptionCode2["Unimplemented"] = "UNIMPLEMENTED";
      ExceptionCode2["Unavailable"] = "UNAVAILABLE";
    })(ExceptionCode || (ExceptionCode = {}));
    CapacitorException = class extends Error {
      constructor(message, code, data) {
        super(message);
        this.message = message;
        this.code = code;
        this.data = data;
      }
    };
    getPlatformId = (win) => {
      var _a, _b;
      if (win === null || win === void 0 ? void 0 : win.androidBridge) {
        return "android";
      } else if ((_b = (_a = win === null || win === void 0 ? void 0 : win.webkit) === null || _a === void 0 ? void 0 : _a.messageHandlers) === null || _b === void 0 ? void 0 : _b.bridge) {
        return "ios";
      } else {
        return "web";
      }
    };
    createCapacitor = (win) => {
      const capCustomPlatform = win.CapacitorCustomPlatform || null;
      const cap = win.Capacitor || {};
      const Plugins = cap.Plugins = cap.Plugins || {};
      const getPlatform = () => {
        return capCustomPlatform !== null ? capCustomPlatform.name : getPlatformId(win);
      };
      const isNativePlatform = () => getPlatform() !== "web";
      const isPluginAvailable = (pluginName) => {
        const plugin = registeredPlugins.get(pluginName);
        if (plugin === null || plugin === void 0 ? void 0 : plugin.platforms.has(getPlatform())) {
          return true;
        }
        if (getPluginHeader(pluginName)) {
          return true;
        }
        return false;
      };
      const getPluginHeader = (pluginName) => {
        var _a;
        return (_a = cap.PluginHeaders) === null || _a === void 0 ? void 0 : _a.find((h) => h.name === pluginName);
      };
      const handleError = (err) => win.console.error(err);
      const registeredPlugins = /* @__PURE__ */ new Map();
      const registerPlugin2 = (pluginName, jsImplementations = {}) => {
        const registeredPlugin = registeredPlugins.get(pluginName);
        if (registeredPlugin) {
          console.warn(`Capacitor plugin "${pluginName}" already registered. Cannot register plugins twice.`);
          return registeredPlugin.proxy;
        }
        const platform = getPlatform();
        const pluginHeader = getPluginHeader(pluginName);
        let jsImplementation;
        const loadPluginImplementation = async () => {
          if (!jsImplementation && platform in jsImplementations) {
            jsImplementation = typeof jsImplementations[platform] === "function" ? jsImplementation = await jsImplementations[platform]() : jsImplementation = jsImplementations[platform];
          } else if (capCustomPlatform !== null && !jsImplementation && "web" in jsImplementations) {
            jsImplementation = typeof jsImplementations["web"] === "function" ? jsImplementation = await jsImplementations["web"]() : jsImplementation = jsImplementations["web"];
          }
          return jsImplementation;
        };
        const createPluginMethod = (impl, prop) => {
          var _a, _b;
          if (pluginHeader) {
            const methodHeader = pluginHeader === null || pluginHeader === void 0 ? void 0 : pluginHeader.methods.find((m) => prop === m.name);
            if (methodHeader) {
              if (methodHeader.rtype === "promise") {
                return (options) => cap.nativePromise(pluginName, prop.toString(), options);
              } else {
                return (options, callback) => cap.nativeCallback(pluginName, prop.toString(), options, callback);
              }
            } else if (impl) {
              return (_a = impl[prop]) === null || _a === void 0 ? void 0 : _a.bind(impl);
            }
          } else if (impl) {
            return (_b = impl[prop]) === null || _b === void 0 ? void 0 : _b.bind(impl);
          } else {
            throw new CapacitorException(`"${pluginName}" plugin is not implemented on ${platform}`, ExceptionCode.Unimplemented);
          }
        };
        const createPluginMethodWrapper = (prop) => {
          let remove;
          const wrapper = (...args) => {
            const p = loadPluginImplementation().then((impl) => {
              const fn = createPluginMethod(impl, prop);
              if (fn) {
                const p2 = fn(...args);
                remove = p2 === null || p2 === void 0 ? void 0 : p2.remove;
                return p2;
              } else {
                throw new CapacitorException(`"${pluginName}.${prop}()" is not implemented on ${platform}`, ExceptionCode.Unimplemented);
              }
            });
            if (prop === "addListener") {
              p.remove = async () => remove();
            }
            return p;
          };
          wrapper.toString = () => `${prop.toString()}() { [capacitor code] }`;
          Object.defineProperty(wrapper, "name", {
            value: prop,
            writable: false,
            configurable: false
          });
          return wrapper;
        };
        const addListener = createPluginMethodWrapper("addListener");
        const removeListener = createPluginMethodWrapper("removeListener");
        const addListenerNative = (eventName, callback) => {
          const call = addListener({ eventName }, callback);
          const remove = async () => {
            const callbackId = await call;
            removeListener({
              eventName,
              callbackId
            }, callback);
          };
          const p = new Promise((resolve) => call.then(() => resolve({ remove })));
          p.remove = async () => {
            console.warn(`Using addListener() without 'await' is deprecated.`);
            await remove();
          };
          return p;
        };
        const proxy = new Proxy({}, {
          get(_, prop) {
            switch (prop) {
              // https://github.com/facebook/react/issues/20030
              case "$$typeof":
                return void 0;
              case "toJSON":
                return () => ({});
              case "addListener":
                return pluginHeader ? addListenerNative : addListener;
              case "removeListener":
                return removeListener;
              default:
                return createPluginMethodWrapper(prop);
            }
          }
        });
        Plugins[pluginName] = proxy;
        registeredPlugins.set(pluginName, {
          name: pluginName,
          proxy,
          platforms: /* @__PURE__ */ new Set([...Object.keys(jsImplementations), ...pluginHeader ? [platform] : []])
        });
        return proxy;
      };
      if (!cap.convertFileSrc) {
        cap.convertFileSrc = (filePath) => filePath;
      }
      cap.getPlatform = getPlatform;
      cap.handleError = handleError;
      cap.isNativePlatform = isNativePlatform;
      cap.isPluginAvailable = isPluginAvailable;
      cap.registerPlugin = registerPlugin2;
      cap.Exception = CapacitorException;
      cap.DEBUG = !!cap.DEBUG;
      cap.isLoggingEnabled = !!cap.isLoggingEnabled;
      return cap;
    };
    initCapacitorGlobal = (win) => win.Capacitor = createCapacitor(win);
    Capacitor = /* @__PURE__ */ initCapacitorGlobal(typeof globalThis !== "undefined" ? globalThis : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : typeof global !== "undefined" ? global : {});
    registerPlugin = Capacitor.registerPlugin;
    WebPlugin = class {
      constructor() {
        this.listeners = {};
        this.retainedEventArguments = {};
        this.windowListeners = {};
      }
      addListener(eventName, listenerFunc) {
        let firstListener = false;
        const listeners = this.listeners[eventName];
        if (!listeners) {
          this.listeners[eventName] = [];
          firstListener = true;
        }
        this.listeners[eventName].push(listenerFunc);
        const windowListener = this.windowListeners[eventName];
        if (windowListener && !windowListener.registered) {
          this.addWindowListener(windowListener);
        }
        if (firstListener) {
          this.sendRetainedArgumentsForEvent(eventName);
        }
        const remove = async () => this.removeListener(eventName, listenerFunc);
        const p = Promise.resolve({ remove });
        return p;
      }
      async removeAllListeners() {
        this.listeners = {};
        for (const listener in this.windowListeners) {
          this.removeWindowListener(this.windowListeners[listener]);
        }
        this.windowListeners = {};
      }
      notifyListeners(eventName, data, retainUntilConsumed) {
        const listeners = this.listeners[eventName];
        if (!listeners) {
          if (retainUntilConsumed) {
            let args = this.retainedEventArguments[eventName];
            if (!args) {
              args = [];
            }
            args.push(data);
            this.retainedEventArguments[eventName] = args;
          }
          return;
        }
        listeners.forEach((listener) => listener(data));
      }
      hasListeners(eventName) {
        var _a;
        return !!((_a = this.listeners[eventName]) === null || _a === void 0 ? void 0 : _a.length);
      }
      registerWindowListener(windowEventName, pluginEventName) {
        this.windowListeners[pluginEventName] = {
          registered: false,
          windowEventName,
          pluginEventName,
          handler: (event) => {
            this.notifyListeners(pluginEventName, event);
          }
        };
      }
      unimplemented(msg = "not implemented") {
        return new Capacitor.Exception(msg, ExceptionCode.Unimplemented);
      }
      unavailable(msg = "not available") {
        return new Capacitor.Exception(msg, ExceptionCode.Unavailable);
      }
      async removeListener(eventName, listenerFunc) {
        const listeners = this.listeners[eventName];
        if (!listeners) {
          return;
        }
        const index = listeners.indexOf(listenerFunc);
        this.listeners[eventName].splice(index, 1);
        if (!this.listeners[eventName].length) {
          this.removeWindowListener(this.windowListeners[eventName]);
        }
      }
      addWindowListener(handle) {
        window.addEventListener(handle.windowEventName, handle.handler);
        handle.registered = true;
      }
      removeWindowListener(handle) {
        if (!handle) {
          return;
        }
        window.removeEventListener(handle.windowEventName, handle.handler);
        handle.registered = false;
      }
      sendRetainedArgumentsForEvent(eventName) {
        const args = this.retainedEventArguments[eventName];
        if (!args) {
          return;
        }
        delete this.retainedEventArguments[eventName];
        args.forEach((arg) => {
          this.notifyListeners(eventName, arg);
        });
      }
    };
    encode = (str) => encodeURIComponent(str).replace(/%(2[346B]|5E|60|7C)/g, decodeURIComponent).replace(/[()]/g, escape);
    decode = (str) => str.replace(/(%[\dA-F]{2})+/gi, decodeURIComponent);
    CapacitorCookiesPluginWeb = class extends WebPlugin {
      async getCookies() {
        const cookies = document.cookie;
        const cookieMap = {};
        cookies.split(";").forEach((cookie) => {
          if (cookie.length <= 0)
            return;
          let [key, value] = cookie.replace(/=/, "CAP_COOKIE").split("CAP_COOKIE");
          key = decode(key).trim();
          value = decode(value).trim();
          cookieMap[key] = value;
        });
        return cookieMap;
      }
      async setCookie(options) {
        try {
          const encodedKey = encode(options.key);
          const encodedValue = encode(options.value);
          const expires = `; expires=${(options.expires || "").replace("expires=", "")}`;
          const path = (options.path || "/").replace("path=", "");
          const domain = options.url != null && options.url.length > 0 ? `domain=${options.url}` : "";
          document.cookie = `${encodedKey}=${encodedValue || ""}${expires}; path=${path}; ${domain};`;
        } catch (error) {
          return Promise.reject(error);
        }
      }
      async deleteCookie(options) {
        try {
          document.cookie = `${options.key}=; Max-Age=0`;
        } catch (error) {
          return Promise.reject(error);
        }
      }
      async clearCookies() {
        try {
          const cookies = document.cookie.split(";") || [];
          for (const cookie of cookies) {
            document.cookie = cookie.replace(/^ +/, "").replace(/=.*/, `=;expires=${(/* @__PURE__ */ new Date()).toUTCString()};path=/`);
          }
        } catch (error) {
          return Promise.reject(error);
        }
      }
      async clearAllCookies() {
        try {
          await this.clearCookies();
        } catch (error) {
          return Promise.reject(error);
        }
      }
    };
    CapacitorCookies = registerPlugin("CapacitorCookies", {
      web: () => new CapacitorCookiesPluginWeb()
    });
    readBlobAsBase64 = async (blob) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64String = reader.result;
        resolve(base64String.indexOf(",") >= 0 ? base64String.split(",")[1] : base64String);
      };
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(blob);
    });
    normalizeHttpHeaders = (headers = {}) => {
      const originalKeys = Object.keys(headers);
      const loweredKeys = Object.keys(headers).map((k) => k.toLocaleLowerCase());
      const normalized = loweredKeys.reduce((acc, key, index) => {
        acc[key] = headers[originalKeys[index]];
        return acc;
      }, {});
      return normalized;
    };
    buildUrlParams = (params, shouldEncode = true) => {
      if (!params)
        return null;
      const output = Object.entries(params).reduce((accumulator, entry) => {
        const [key, value] = entry;
        let encodedValue;
        let item;
        if (Array.isArray(value)) {
          item = "";
          value.forEach((str) => {
            encodedValue = shouldEncode ? encodeURIComponent(str) : str;
            item += `${key}=${encodedValue}&`;
          });
          item.slice(0, -1);
        } else {
          encodedValue = shouldEncode ? encodeURIComponent(value) : value;
          item = `${key}=${encodedValue}`;
        }
        return `${accumulator}&${item}`;
      }, "");
      return output.substr(1);
    };
    buildRequestInit = (options, extra = {}) => {
      const output = Object.assign({ method: options.method || "GET", headers: options.headers }, extra);
      const headers = normalizeHttpHeaders(options.headers);
      const type = headers["content-type"] || "";
      if (typeof options.data === "string") {
        output.body = options.data;
      } else if (type.includes("application/x-www-form-urlencoded")) {
        const params = new URLSearchParams();
        for (const [key, value] of Object.entries(options.data || {})) {
          params.set(key, value);
        }
        output.body = params.toString();
      } else if (type.includes("multipart/form-data") || options.data instanceof FormData) {
        const form = new FormData();
        if (options.data instanceof FormData) {
          options.data.forEach((value, key) => {
            form.append(key, value);
          });
        } else {
          for (const key of Object.keys(options.data)) {
            form.append(key, options.data[key]);
          }
        }
        output.body = form;
        const headers2 = new Headers(output.headers);
        headers2.delete("content-type");
        output.headers = headers2;
      } else if (type.includes("application/json") || typeof options.data === "object") {
        output.body = JSON.stringify(options.data);
      }
      return output;
    };
    CapacitorHttpPluginWeb = class extends WebPlugin {
      /**
       * Perform an Http request given a set of options
       * @param options Options to build the HTTP request
       */
      async request(options) {
        const requestInit = buildRequestInit(options, options.webFetchExtra);
        const urlParams = buildUrlParams(options.params, options.shouldEncodeUrlParams);
        const url = urlParams ? `${options.url}?${urlParams}` : options.url;
        const response = await fetch(url, requestInit);
        const contentType = response.headers.get("content-type") || "";
        let { responseType = "text" } = response.ok ? options : {};
        if (contentType.includes("application/json")) {
          responseType = "json";
        }
        let data;
        let blob;
        switch (responseType) {
          case "arraybuffer":
          case "blob":
            blob = await response.blob();
            data = await readBlobAsBase64(blob);
            break;
          case "json":
            data = await response.json();
            break;
          case "document":
          case "text":
          default:
            data = await response.text();
        }
        const headers = {};
        response.headers.forEach((value, key) => {
          headers[key] = value;
        });
        return {
          data,
          headers,
          status: response.status,
          url: response.url
        };
      }
      /**
       * Perform an Http GET request given a set of options
       * @param options Options to build the HTTP request
       */
      async get(options) {
        return this.request(Object.assign(Object.assign({}, options), { method: "GET" }));
      }
      /**
       * Perform an Http POST request given a set of options
       * @param options Options to build the HTTP request
       */
      async post(options) {
        return this.request(Object.assign(Object.assign({}, options), { method: "POST" }));
      }
      /**
       * Perform an Http PUT request given a set of options
       * @param options Options to build the HTTP request
       */
      async put(options) {
        return this.request(Object.assign(Object.assign({}, options), { method: "PUT" }));
      }
      /**
       * Perform an Http PATCH request given a set of options
       * @param options Options to build the HTTP request
       */
      async patch(options) {
        return this.request(Object.assign(Object.assign({}, options), { method: "PATCH" }));
      }
      /**
       * Perform an Http DELETE request given a set of options
       * @param options Options to build the HTTP request
       */
      async delete(options) {
        return this.request(Object.assign(Object.assign({}, options), { method: "DELETE" }));
      }
    };
    CapacitorHttp = registerPlugin("CapacitorHttp", {
      web: () => new CapacitorHttpPluginWeb()
    });
    (function(SystemBarsStyle2) {
      SystemBarsStyle2["Dark"] = "DARK";
      SystemBarsStyle2["Light"] = "LIGHT";
      SystemBarsStyle2["Default"] = "DEFAULT";
    })(SystemBarsStyle || (SystemBarsStyle = {}));
    (function(SystemBarType2) {
      SystemBarType2["StatusBar"] = "StatusBar";
      SystemBarType2["NavigationBar"] = "NavigationBar";
    })(SystemBarType || (SystemBarType = {}));
    SystemBarsPluginWeb = class extends WebPlugin {
      async setStyle() {
        this.unavailable("not available for web");
      }
      async setAnimation() {
        this.unavailable("not available for web");
      }
      async show() {
        this.unavailable("not available for web");
      }
      async hide() {
        this.unavailable("not available for web");
      }
    };
    SystemBars = registerPlugin("SystemBars", {
      web: () => new SystemBarsPluginWeb()
    });
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/generated/error-codes.js
var PURCHASES_ERROR_CODE;
var init_error_codes = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/generated/error-codes.js"() {
    (function(PURCHASES_ERROR_CODE2) {
      PURCHASES_ERROR_CODE2["UNKNOWN_ERROR"] = "0";
      PURCHASES_ERROR_CODE2["PURCHASE_CANCELLED_ERROR"] = "1";
      PURCHASES_ERROR_CODE2["STORE_PROBLEM_ERROR"] = "2";
      PURCHASES_ERROR_CODE2["PURCHASE_NOT_ALLOWED_ERROR"] = "3";
      PURCHASES_ERROR_CODE2["PURCHASE_INVALID_ERROR"] = "4";
      PURCHASES_ERROR_CODE2["PRODUCT_NOT_AVAILABLE_FOR_PURCHASE_ERROR"] = "5";
      PURCHASES_ERROR_CODE2["PRODUCT_ALREADY_PURCHASED_ERROR"] = "6";
      PURCHASES_ERROR_CODE2["RECEIPT_ALREADY_IN_USE_ERROR"] = "7";
      PURCHASES_ERROR_CODE2["INVALID_RECEIPT_ERROR"] = "8";
      PURCHASES_ERROR_CODE2["MISSING_RECEIPT_FILE_ERROR"] = "9";
      PURCHASES_ERROR_CODE2["NETWORK_ERROR"] = "10";
      PURCHASES_ERROR_CODE2["INVALID_CREDENTIALS_ERROR"] = "11";
      PURCHASES_ERROR_CODE2["UNEXPECTED_BACKEND_RESPONSE_ERROR"] = "12";
      PURCHASES_ERROR_CODE2["RECEIPT_IN_USE_BY_OTHER_SUBSCRIBER_ERROR"] = "13";
      PURCHASES_ERROR_CODE2["INVALID_APP_USER_ID_ERROR"] = "14";
      PURCHASES_ERROR_CODE2["OPERATION_ALREADY_IN_PROGRESS_ERROR"] = "15";
      PURCHASES_ERROR_CODE2["UNKNOWN_BACKEND_ERROR"] = "16";
      PURCHASES_ERROR_CODE2["INVALID_APPLE_SUBSCRIPTION_KEY_ERROR"] = "17";
      PURCHASES_ERROR_CODE2["INELIGIBLE_ERROR"] = "18";
      PURCHASES_ERROR_CODE2["INSUFFICIENT_PERMISSIONS_ERROR"] = "19";
      PURCHASES_ERROR_CODE2["PAYMENT_PENDING_ERROR"] = "20";
      PURCHASES_ERROR_CODE2["INVALID_SUBSCRIBER_ATTRIBUTES_ERROR"] = "21";
      PURCHASES_ERROR_CODE2["LOG_OUT_ANONYMOUS_USER_ERROR"] = "22";
      PURCHASES_ERROR_CODE2["CONFIGURATION_ERROR"] = "23";
      PURCHASES_ERROR_CODE2["UNSUPPORTED_ERROR"] = "24";
      PURCHASES_ERROR_CODE2["EMPTY_SUBSCRIBER_ATTRIBUTES_ERROR"] = "25";
      PURCHASES_ERROR_CODE2["PRODUCT_DISCOUNT_MISSING_IDENTIFIER_ERROR"] = "26";
      PURCHASES_ERROR_CODE2["PRODUCT_DISCOUNT_MISSING_SUBSCRIPTION_GROUP_IDENTIFIER_ERROR"] = "28";
      PURCHASES_ERROR_CODE2["CUSTOMER_INFO_ERROR"] = "29";
      PURCHASES_ERROR_CODE2["SYSTEM_INFO_ERROR"] = "30";
      PURCHASES_ERROR_CODE2["BEGIN_REFUND_REQUEST_ERROR"] = "31";
      PURCHASES_ERROR_CODE2["PRODUCT_REQUEST_TIMED_OUT_ERROR"] = "32";
      PURCHASES_ERROR_CODE2["API_ENDPOINT_BLOCKED"] = "33";
      PURCHASES_ERROR_CODE2["INVALID_PROMOTIONAL_OFFER_ERROR"] = "34";
      PURCHASES_ERROR_CODE2["OFFLINE_CONNECTION_ERROR"] = "35";
      PURCHASES_ERROR_CODE2["FEATURE_NOT_AVAILABLE_IN_CUSTOM_ENTITLEMENTS_COMPUTATION_MODE"] = "36";
      PURCHASES_ERROR_CODE2["SIGNATURE_VERIFICATION_ERROR"] = "37";
      PURCHASES_ERROR_CODE2["FEATURE_NOT_SUPPORTED_WITH_STORE_KIT_1"] = "38";
      PURCHASES_ERROR_CODE2["INVALID_WEB_PURCHASE_TOKEN"] = "39";
      PURCHASES_ERROR_CODE2["PURCHASE_BELONGS_TO_OTHER_USER"] = "40";
      PURCHASES_ERROR_CODE2["EXPIRED_WEB_PURCHASE_TOKEN"] = "41";
      PURCHASES_ERROR_CODE2["TEST_STORE_SIMULATED_PURCHASE_ERROR"] = "42";
    })(PURCHASES_ERROR_CODE || (PURCHASES_ERROR_CODE = {}));
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/errors.js
var __extends, UninitializedPurchasesError, UnsupportedPlatformError;
var init_errors = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/errors.js"() {
    init_error_codes();
    __extends = /* @__PURE__ */ (function() {
      var extendStatics = function(d, b) {
        extendStatics = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(d2, b2) {
          d2.__proto__ = b2;
        } || function(d2, b2) {
          for (var p in b2) if (Object.prototype.hasOwnProperty.call(b2, p)) d2[p] = b2[p];
        };
        return extendStatics(d, b);
      };
      return function(d, b) {
        if (typeof b !== "function" && b !== null)
          throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() {
          this.constructor = d;
        }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
      };
    })();
    UninitializedPurchasesError = /** @class */
    (function(_super) {
      __extends(UninitializedPurchasesError2, _super);
      function UninitializedPurchasesError2() {
        var _this = _super.call(this, "There is no singleton instance. Make sure you configure Purchases before trying to get the default instance. More info here: https://errors.rev.cat/configuring-sdk") || this;
        Object.setPrototypeOf(_this, UninitializedPurchasesError2.prototype);
        return _this;
      }
      return UninitializedPurchasesError2;
    })(Error);
    UnsupportedPlatformError = /** @class */
    (function(_super) {
      __extends(UnsupportedPlatformError2, _super);
      function UnsupportedPlatformError2() {
        var _this = _super.call(this, "This method is not available in the current platform.") || this;
        Object.setPrototypeOf(_this, UnsupportedPlatformError2.prototype);
        return _this;
      }
      return UnsupportedPlatformError2;
    })(Error);
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/errorNormalizer.js
function isRecord(value) {
  return typeof value === "object" && value !== null;
}
function readPayload(error) {
  for (var _i = 0, PAYLOAD_KEYS_1 = PAYLOAD_KEYS; _i < PAYLOAD_KEYS_1.length; _i++) {
    var key = PAYLOAD_KEYS_1[_i];
    var candidate = error[key];
    if (isRecord(candidate)) {
      return candidate;
    }
  }
  return {};
}
function firstString() {
  var values = [];
  for (var _i = 0; _i < arguments.length; _i++) {
    values[_i] = arguments[_i];
  }
  for (var _a = 0, values_1 = values; _a < values_1.length; _a++) {
    var value = values_1[_a];
    if (typeof value === "string") {
      return value;
    }
  }
  return "";
}
function readCode(error, payload) {
  var _a;
  var candidate = (_a = error.code) !== null && _a !== void 0 ? _a : payload.code;
  var code = String(candidate);
  return /^\d+$/.test(code) ? code : void 0;
}
function normalizePurchasesError(error) {
  if (!isRecord(error)) {
    return error;
  }
  var payload = readPayload(error);
  var code = readCode(error, payload);
  if (code === void 0) {
    return error;
  }
  error.code = code;
  var userInfo = __assign({}, payload);
  if (typeof userInfo.readableErrorCode !== "string") {
    userInfo.readableErrorCode = firstString(payload.readableErrorCode, error.readableErrorCode);
  }
  error.userInfo = userInfo;
  if (typeof error.message !== "string" || error.message === "") {
    error.message = firstString(payload.message);
  }
  if (typeof error.readableErrorCode !== "string") {
    error.readableErrorCode = userInfo.readableErrorCode;
  }
  if (typeof error.underlyingErrorMessage !== "string") {
    error.underlyingErrorMessage = firstString(payload.underlyingErrorMessage);
  }
  error.userCancelled = code === PURCHASES_ERROR_CODE.PURCHASE_CANCELLED_ERROR;
  return error;
}
var __assign, PAYLOAD_KEYS;
var init_errorNormalizer = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/errorNormalizer.js"() {
    init_error_codes();
    __assign = function() {
      __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
          s = arguments[i];
          for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
            t[p] = s[p];
        }
        return t;
      };
      return __assign.apply(this, arguments);
    };
    PAYLOAD_KEYS = ["userInfo", "data", "info"];
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/customerInfo.js
var init_customerInfo = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/customerInfo.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/offerings.js
var PACKAGE_TYPE, INTRO_ELIGIBILITY_STATUS, PRODUCT_CATEGORY, PRODUCT_TYPE, PRORATION_MODE, STORE_REPLACEMENT_MODE, BILLING_PLAN_TYPE, RECURRENCE_MODE, OFFER_PAYMENT_MODE, PERIOD_UNIT;
var init_offerings = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/offerings.js"() {
    (function(PACKAGE_TYPE2) {
      PACKAGE_TYPE2["UNKNOWN"] = "UNKNOWN";
      PACKAGE_TYPE2["CUSTOM"] = "CUSTOM";
      PACKAGE_TYPE2["LIFETIME"] = "LIFETIME";
      PACKAGE_TYPE2["ANNUAL"] = "ANNUAL";
      PACKAGE_TYPE2["SIX_MONTH"] = "SIX_MONTH";
      PACKAGE_TYPE2["THREE_MONTH"] = "THREE_MONTH";
      PACKAGE_TYPE2["TWO_MONTH"] = "TWO_MONTH";
      PACKAGE_TYPE2["MONTHLY"] = "MONTHLY";
      PACKAGE_TYPE2["WEEKLY"] = "WEEKLY";
    })(PACKAGE_TYPE || (PACKAGE_TYPE = {}));
    (function(INTRO_ELIGIBILITY_STATUS2) {
      INTRO_ELIGIBILITY_STATUS2[INTRO_ELIGIBILITY_STATUS2["INTRO_ELIGIBILITY_STATUS_UNKNOWN"] = 0] = "INTRO_ELIGIBILITY_STATUS_UNKNOWN";
      INTRO_ELIGIBILITY_STATUS2[INTRO_ELIGIBILITY_STATUS2["INTRO_ELIGIBILITY_STATUS_INELIGIBLE"] = 1] = "INTRO_ELIGIBILITY_STATUS_INELIGIBLE";
      INTRO_ELIGIBILITY_STATUS2[INTRO_ELIGIBILITY_STATUS2["INTRO_ELIGIBILITY_STATUS_ELIGIBLE"] = 2] = "INTRO_ELIGIBILITY_STATUS_ELIGIBLE";
      INTRO_ELIGIBILITY_STATUS2[INTRO_ELIGIBILITY_STATUS2["INTRO_ELIGIBILITY_STATUS_NO_INTRO_OFFER_EXISTS"] = 3] = "INTRO_ELIGIBILITY_STATUS_NO_INTRO_OFFER_EXISTS";
    })(INTRO_ELIGIBILITY_STATUS || (INTRO_ELIGIBILITY_STATUS = {}));
    (function(PRODUCT_CATEGORY2) {
      PRODUCT_CATEGORY2["NON_SUBSCRIPTION"] = "NON_SUBSCRIPTION";
      PRODUCT_CATEGORY2["SUBSCRIPTION"] = "SUBSCRIPTION";
      PRODUCT_CATEGORY2["UNKNOWN"] = "UNKNOWN";
    })(PRODUCT_CATEGORY || (PRODUCT_CATEGORY = {}));
    (function(PRODUCT_TYPE2) {
      PRODUCT_TYPE2["CONSUMABLE"] = "CONSUMABLE";
      PRODUCT_TYPE2["NON_CONSUMABLE"] = "NON_CONSUMABLE";
      PRODUCT_TYPE2["NON_RENEWABLE_SUBSCRIPTION"] = "NON_RENEWABLE_SUBSCRIPTION";
      PRODUCT_TYPE2["AUTO_RENEWABLE_SUBSCRIPTION"] = "AUTO_RENEWABLE_SUBSCRIPTION";
      PRODUCT_TYPE2["PREPAID_SUBSCRIPTION"] = "PREPAID_SUBSCRIPTION";
      PRODUCT_TYPE2["UNKNOWN"] = "UNKNOWN";
    })(PRODUCT_TYPE || (PRODUCT_TYPE = {}));
    (function(PRORATION_MODE2) {
      PRORATION_MODE2[PRORATION_MODE2["UNKNOWN_SUBSCRIPTION_UPGRADE_DOWNGRADE_POLICY"] = 0] = "UNKNOWN_SUBSCRIPTION_UPGRADE_DOWNGRADE_POLICY";
      PRORATION_MODE2[PRORATION_MODE2["IMMEDIATE_WITH_TIME_PRORATION"] = 1] = "IMMEDIATE_WITH_TIME_PRORATION";
      PRORATION_MODE2[PRORATION_MODE2["IMMEDIATE_AND_CHARGE_PRORATED_PRICE"] = 2] = "IMMEDIATE_AND_CHARGE_PRORATED_PRICE";
      PRORATION_MODE2[PRORATION_MODE2["IMMEDIATE_WITHOUT_PRORATION"] = 3] = "IMMEDIATE_WITHOUT_PRORATION";
      PRORATION_MODE2[PRORATION_MODE2["DEFERRED"] = 6] = "DEFERRED";
      PRORATION_MODE2[PRORATION_MODE2["IMMEDIATE_AND_CHARGE_FULL_PRICE"] = 5] = "IMMEDIATE_AND_CHARGE_FULL_PRICE";
    })(PRORATION_MODE || (PRORATION_MODE = {}));
    (function(STORE_REPLACEMENT_MODE2) {
      STORE_REPLACEMENT_MODE2["WITHOUT_PRORATION"] = "WITHOUT_PRORATION";
      STORE_REPLACEMENT_MODE2["WITH_TIME_PRORATION"] = "WITH_TIME_PRORATION";
      STORE_REPLACEMENT_MODE2["CHARGE_FULL_PRICE"] = "CHARGE_FULL_PRICE";
      STORE_REPLACEMENT_MODE2["CHARGE_PRORATED_PRICE"] = "CHARGE_PRORATED_PRICE";
      STORE_REPLACEMENT_MODE2["DEFERRED"] = "DEFERRED";
    })(STORE_REPLACEMENT_MODE || (STORE_REPLACEMENT_MODE = {}));
    (function(BILLING_PLAN_TYPE2) {
      BILLING_PLAN_TYPE2["UP_FRONT"] = "UP_FRONT";
      BILLING_PLAN_TYPE2["MONTHLY"] = "MONTHLY";
      BILLING_PLAN_TYPE2["UNKNOWN"] = "UNKNOWN";
    })(BILLING_PLAN_TYPE || (BILLING_PLAN_TYPE = {}));
    (function(RECURRENCE_MODE2) {
      RECURRENCE_MODE2[RECURRENCE_MODE2["INFINITE_RECURRING"] = 1] = "INFINITE_RECURRING";
      RECURRENCE_MODE2[RECURRENCE_MODE2["FINITE_RECURRING"] = 2] = "FINITE_RECURRING";
      RECURRENCE_MODE2[RECURRENCE_MODE2["NON_RECURRING"] = 3] = "NON_RECURRING";
    })(RECURRENCE_MODE || (RECURRENCE_MODE = {}));
    (function(OFFER_PAYMENT_MODE2) {
      OFFER_PAYMENT_MODE2["FREE_TRIAL"] = "FREE_TRIAL";
      OFFER_PAYMENT_MODE2["SINGLE_PAYMENT"] = "SINGLE_PAYMENT";
      OFFER_PAYMENT_MODE2["DISCOUNTED_RECURRING_PAYMENT"] = "DISCOUNTED_RECURRING_PAYMENT";
    })(OFFER_PAYMENT_MODE || (OFFER_PAYMENT_MODE = {}));
    (function(PERIOD_UNIT2) {
      PERIOD_UNIT2["DAY"] = "DAY";
      PERIOD_UNIT2["WEEK"] = "WEEK";
      PERIOD_UNIT2["MONTH"] = "MONTH";
      PERIOD_UNIT2["YEAR"] = "YEAR";
      PERIOD_UNIT2["UNKNOWN"] = "UNKNOWN";
    })(PERIOD_UNIT || (PERIOD_UNIT = {}));
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/enums.js
var PURCHASE_TYPE, BILLING_FEATURE, REFUND_REQUEST_STATUS, LOG_LEVEL, IN_APP_MESSAGE_TYPE, ENTITLEMENT_VERIFICATION_MODE, VERIFICATION_RESULT, PAYWALL_RESULT, STOREKIT_VERSION, PURCHASES_ARE_COMPLETED_BY_TYPE;
var init_enums = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/enums.js"() {
    (function(PURCHASE_TYPE2) {
      PURCHASE_TYPE2["INAPP"] = "inapp";
      PURCHASE_TYPE2["SUBS"] = "subs";
    })(PURCHASE_TYPE || (PURCHASE_TYPE = {}));
    (function(BILLING_FEATURE2) {
      BILLING_FEATURE2[BILLING_FEATURE2["SUBSCRIPTIONS"] = 0] = "SUBSCRIPTIONS";
      BILLING_FEATURE2[BILLING_FEATURE2["SUBSCRIPTIONS_UPDATE"] = 1] = "SUBSCRIPTIONS_UPDATE";
      BILLING_FEATURE2[BILLING_FEATURE2["IN_APP_ITEMS_ON_VR"] = 2] = "IN_APP_ITEMS_ON_VR";
      BILLING_FEATURE2[BILLING_FEATURE2["SUBSCRIPTIONS_ON_VR"] = 3] = "SUBSCRIPTIONS_ON_VR";
      BILLING_FEATURE2[BILLING_FEATURE2["PRICE_CHANGE_CONFIRMATION"] = 4] = "PRICE_CHANGE_CONFIRMATION";
    })(BILLING_FEATURE || (BILLING_FEATURE = {}));
    (function(REFUND_REQUEST_STATUS2) {
      REFUND_REQUEST_STATUS2[REFUND_REQUEST_STATUS2["SUCCESS"] = 0] = "SUCCESS";
      REFUND_REQUEST_STATUS2[REFUND_REQUEST_STATUS2["USER_CANCELLED"] = 1] = "USER_CANCELLED";
      REFUND_REQUEST_STATUS2[REFUND_REQUEST_STATUS2["ERROR"] = 2] = "ERROR";
    })(REFUND_REQUEST_STATUS || (REFUND_REQUEST_STATUS = {}));
    (function(LOG_LEVEL2) {
      LOG_LEVEL2["VERBOSE"] = "VERBOSE";
      LOG_LEVEL2["DEBUG"] = "DEBUG";
      LOG_LEVEL2["INFO"] = "INFO";
      LOG_LEVEL2["WARN"] = "WARN";
      LOG_LEVEL2["ERROR"] = "ERROR";
    })(LOG_LEVEL || (LOG_LEVEL = {}));
    (function(IN_APP_MESSAGE_TYPE2) {
      IN_APP_MESSAGE_TYPE2[IN_APP_MESSAGE_TYPE2["BILLING_ISSUE"] = 0] = "BILLING_ISSUE";
      IN_APP_MESSAGE_TYPE2[IN_APP_MESSAGE_TYPE2["PRICE_INCREASE_CONSENT"] = 1] = "PRICE_INCREASE_CONSENT";
      IN_APP_MESSAGE_TYPE2[IN_APP_MESSAGE_TYPE2["GENERIC"] = 2] = "GENERIC";
      IN_APP_MESSAGE_TYPE2[IN_APP_MESSAGE_TYPE2["WIN_BACK_OFFER"] = 3] = "WIN_BACK_OFFER";
    })(IN_APP_MESSAGE_TYPE || (IN_APP_MESSAGE_TYPE = {}));
    (function(ENTITLEMENT_VERIFICATION_MODE2) {
      ENTITLEMENT_VERIFICATION_MODE2["DISABLED"] = "DISABLED";
      ENTITLEMENT_VERIFICATION_MODE2["INFORMATIONAL"] = "INFORMATIONAL";
    })(ENTITLEMENT_VERIFICATION_MODE || (ENTITLEMENT_VERIFICATION_MODE = {}));
    (function(VERIFICATION_RESULT2) {
      VERIFICATION_RESULT2["NOT_REQUESTED"] = "NOT_REQUESTED";
      VERIFICATION_RESULT2["VERIFIED"] = "VERIFIED";
      VERIFICATION_RESULT2["FAILED"] = "FAILED";
      VERIFICATION_RESULT2["VERIFIED_ON_DEVICE"] = "VERIFIED_ON_DEVICE";
    })(VERIFICATION_RESULT || (VERIFICATION_RESULT = {}));
    (function(PAYWALL_RESULT2) {
      PAYWALL_RESULT2["NOT_PRESENTED"] = "NOT_PRESENTED";
      PAYWALL_RESULT2["ERROR"] = "ERROR";
      PAYWALL_RESULT2["CANCELLED"] = "CANCELLED";
      PAYWALL_RESULT2["PURCHASED"] = "PURCHASED";
      PAYWALL_RESULT2["RESTORED"] = "RESTORED";
    })(PAYWALL_RESULT || (PAYWALL_RESULT = {}));
    (function(STOREKIT_VERSION2) {
      STOREKIT_VERSION2["STOREKIT_1"] = "STOREKIT_1";
      STOREKIT_VERSION2["STOREKIT_2"] = "STOREKIT_2";
      STOREKIT_VERSION2["DEFAULT"] = "DEFAULT";
    })(STOREKIT_VERSION || (STOREKIT_VERSION = {}));
    (function(PURCHASES_ARE_COMPLETED_BY_TYPE2) {
      PURCHASES_ARE_COMPLETED_BY_TYPE2["MY_APP"] = "MY_APP";
      PURCHASES_ARE_COMPLETED_BY_TYPE2["REVENUECAT"] = "REVENUECAT";
    })(PURCHASES_ARE_COMPLETED_BY_TYPE || (PURCHASES_ARE_COMPLETED_BY_TYPE = {}));
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/purchaseParams.js
var init_purchaseParams = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/purchaseParams.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/purchasesConfiguration.js
var init_purchasesConfiguration = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/purchasesConfiguration.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/callbackTypes.js
var init_callbackTypes = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/callbackTypes.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/webRedemption.js
var WebPurchaseRedemptionResultType;
var init_webRedemption = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/webRedemption.js"() {
    (function(WebPurchaseRedemptionResultType2) {
      WebPurchaseRedemptionResultType2["SUCCESS"] = "SUCCESS";
      WebPurchaseRedemptionResultType2["ERROR"] = "ERROR";
      WebPurchaseRedemptionResultType2["PURCHASE_BELONGS_TO_OTHER_USER"] = "PURCHASE_BELONGS_TO_OTHER_USER";
      WebPurchaseRedemptionResultType2["INVALID_TOKEN"] = "INVALID_TOKEN";
      WebPurchaseRedemptionResultType2["EXPIRED"] = "EXPIRED";
    })(WebPurchaseRedemptionResultType || (WebPurchaseRedemptionResultType = {}));
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/storefront.js
var init_storefront = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/storefront.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/virtualCurrency.js
var init_virtualCurrency = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/virtualCurrency.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/paywallInteractionEvent.js
var init_paywallInteractionEvent = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/paywallInteractionEvent.js"() {
  }
});

// node_modules/@revenuecat/purchases-typescript-internal-esm/dist/index.js
var init_dist2 = __esm({
  "node_modules/@revenuecat/purchases-typescript-internal-esm/dist/index.js"() {
    init_errors();
    init_errorNormalizer();
    init_customerInfo();
    init_offerings();
    init_enums();
    init_purchaseParams();
    init_purchasesConfiguration();
    init_callbackTypes();
    init_webRedemption();
    init_storefront();
    init_virtualCurrency();
    init_paywallInteractionEvent();
  }
});

// node_modules/@revenuecat/purchases-capacitor/dist/esm/web.js
var web_exports = {};
__export(web_exports, {
  PurchasesWeb: () => PurchasesWeb
});
var PurchasesWeb;
var init_web = __esm({
  "node_modules/@revenuecat/purchases-capacitor/dist/esm/web.js"() {
    init_dist();
    init_dist2();
    PurchasesWeb = class extends WebPlugin {
      constructor() {
        super(...arguments);
        this.shouldMockWebResults = false;
        this.webNotSupportedErrorMessage = "Web not supported in this plugin.";
        this.mockEmptyCustomerInfo = {
          entitlements: {
            all: {},
            active: {},
            verification: VERIFICATION_RESULT.NOT_REQUESTED
          },
          activeSubscriptions: [],
          allPurchasedProductIdentifiers: [],
          latestExpirationDate: null,
          firstSeen: "2023-08-31T15:11:21.445Z",
          originalAppUserId: "mock-web-user-id",
          requestDate: "2023-08-31T15:11:21.445Z",
          allExpirationDates: {},
          allPurchaseDates: {},
          originalApplicationVersion: null,
          originalPurchaseDate: null,
          managementURL: null,
          nonSubscriptionTransactions: [],
          subscriptionsByProductIdentifier: {}
        };
        this.mockEmptyVirtualCurrencies = {
          all: {}
        };
      }
      configure(_configuration) {
        return this.mockNonReturningFunctionIfEnabled("configure");
      }
      parseAsWebPurchaseRedemption(_options) {
        return this.mockReturningFunctionIfEnabled("parseAsWebPurchaseRedemption", { webPurchaseRedemption: null });
      }
      redeemWebPurchase(_options) {
        return this.mockReturningFunctionIfEnabled("redeemWebPurchase", {
          result: WebPurchaseRedemptionResultType.INVALID_TOKEN
        });
      }
      setMockWebResults(options) {
        this.shouldMockWebResults = options.shouldMockWebResults;
        return Promise.resolve();
      }
      setSimulatesAskToBuyInSandbox(_simulatesAskToBuyInSandbox) {
        return this.mockNonReturningFunctionIfEnabled("setSimulatesAskToBuyInSandbox");
      }
      addCustomerInfoUpdateListener(_customerInfoUpdateListener) {
        return this.mockReturningFunctionIfEnabled("addCustomerInfoUpdateListener", "mock-callback-id");
      }
      removeCustomerInfoUpdateListener(_options) {
        return this.mockReturningFunctionIfEnabled("removeCustomerInfoUpdateListener", { wasRemoved: false });
      }
      addShouldPurchasePromoProductListener(_shouldPurchasePromoProductListener) {
        return this.mockReturningFunctionIfEnabled("addShouldPurchasePromoProductListener", "mock-callback-id");
      }
      removeShouldPurchasePromoProductListener(_listenerToRemove) {
        return this.mockReturningFunctionIfEnabled("removeShouldPurchasePromoProductListener", { wasRemoved: false });
      }
      getOfferings() {
        const mockOfferings = {
          all: {},
          current: null
        };
        return this.mockReturningFunctionIfEnabled("getOfferings", mockOfferings);
      }
      getCurrentOfferingForPlacement(_options) {
        const mockOffering = null;
        return this.mockReturningFunctionIfEnabled("getCurrentOfferingForPlacement", mockOffering);
      }
      syncAttributesAndOfferingsIfNeeded() {
        const mockOfferings = {
          all: {},
          current: null
        };
        return this.mockReturningFunctionIfEnabled("syncAttributesAndOfferingsIfNeeded", mockOfferings);
      }
      getProducts(_options) {
        const mockProducts = { products: [] };
        return this.mockReturningFunctionIfEnabled("getProducts", mockProducts);
      }
      purchaseStoreProduct(_options) {
        const mockPurchaseResult = {
          productIdentifier: _options.product.identifier,
          customerInfo: this.mockEmptyCustomerInfo,
          transaction: this.mockTransaction(_options.product.identifier)
        };
        return this.mockReturningFunctionIfEnabled("purchaseStoreProduct", mockPurchaseResult);
      }
      purchaseDiscountedProduct(_options) {
        const mockPurchaseResult = {
          productIdentifier: _options.product.identifier,
          customerInfo: this.mockEmptyCustomerInfo,
          transaction: this.mockTransaction(_options.product.identifier)
        };
        return this.mockReturningFunctionIfEnabled("purchaseDiscountedProduct", mockPurchaseResult);
      }
      purchasePackage(_options) {
        const mockPurchaseResult = {
          productIdentifier: _options.aPackage.product.identifier,
          customerInfo: this.mockEmptyCustomerInfo,
          transaction: this.mockTransaction(_options.aPackage.product.identifier)
        };
        return this.mockReturningFunctionIfEnabled("purchasePackage", mockPurchaseResult);
      }
      purchaseSubscriptionOption(_options) {
        const mockPurchaseResult = {
          productIdentifier: _options.subscriptionOption.productId,
          customerInfo: this.mockEmptyCustomerInfo,
          transaction: this.mockTransaction(_options.subscriptionOption.productId)
        };
        return this.mockReturningFunctionIfEnabled("purchaseSubscriptionOption", mockPurchaseResult);
      }
      purchaseDiscountedPackage(_options) {
        const mockPurchaseResult = {
          productIdentifier: _options.aPackage.product.identifier,
          customerInfo: this.mockEmptyCustomerInfo,
          transaction: this.mockTransaction(_options.aPackage.product.identifier)
        };
        return this.mockReturningFunctionIfEnabled("purchaseDiscountedPackage", mockPurchaseResult);
      }
      restorePurchases() {
        const mockResponse = { customerInfo: this.mockEmptyCustomerInfo };
        return this.mockReturningFunctionIfEnabled("restorePurchases", mockResponse);
      }
      recordPurchase(options) {
        const mockResponse = {
          transaction: this.mockTransaction(options.productID)
        };
        return this.mockReturningFunctionIfEnabled("recordPurchase", mockResponse);
      }
      getAppUserID() {
        return this.mockReturningFunctionIfEnabled("getAppUserID", {
          appUserID: "test-web-user-id"
        });
      }
      getStorefront() {
        return this.mockReturningFunctionIfEnabled("getStorefront", {
          countryCode: "USA"
        });
      }
      logIn(_appUserID) {
        const mockLogInResult = {
          customerInfo: this.mockEmptyCustomerInfo,
          created: false
        };
        return this.mockReturningFunctionIfEnabled("logIn", mockLogInResult);
      }
      logOut() {
        const mockResponse = { customerInfo: this.mockEmptyCustomerInfo };
        return this.mockReturningFunctionIfEnabled("logOut", mockResponse);
      }
      setLogLevel(_level) {
        return this.mockNonReturningFunctionIfEnabled("setLogLevel");
      }
      setLogHandler(_logHandler) {
        return this.mockNonReturningFunctionIfEnabled("setLogHandler");
      }
      getCustomerInfo() {
        const mockResponse = { customerInfo: this.mockEmptyCustomerInfo };
        return this.mockReturningFunctionIfEnabled("getCustomerInfo", mockResponse);
      }
      syncPurchases() {
        return this.mockNonReturningFunctionIfEnabled("syncPurchases");
      }
      syncObserverModeAmazonPurchase(_options) {
        return this.mockNonReturningFunctionIfEnabled("syncObserverModeAmazonPurchase");
      }
      syncAmazonPurchase(_options) {
        return this.mockNonReturningFunctionIfEnabled("syncAmazonPurchase");
      }
      enableAdServicesAttributionTokenCollection() {
        return this.mockNonReturningFunctionIfEnabled("enableAdServicesAttributionTokenCollection");
      }
      isAnonymous() {
        const mockResponse = { isAnonymous: false };
        return this.mockReturningFunctionIfEnabled("isAnonymous", mockResponse);
      }
      checkTrialOrIntroductoryPriceEligibility(_productIdentifiers) {
        return this.mockReturningFunctionIfEnabled("checkTrialOrIntroductoryPriceEligibility", {});
      }
      getPromotionalOffer(_options) {
        return this.mockReturningFunctionIfEnabled("getPromotionalOffer", void 0);
      }
      getEligibleWinBackOffersForProduct(_options) {
        return this.mockReturningFunctionIfEnabled("getEligibleWinBackOffersForProduct", { eligibleWinBackOffers: [] });
      }
      getEligibleWinBackOffersForPackage(_options) {
        return this.mockReturningFunctionIfEnabled("getEligibleWinBackOffersForPackage", { eligibleWinBackOffers: [] });
      }
      purchaseProductWithWinBackOffer(_options) {
        return this.mockReturningFunctionIfEnabled("purchaseProductWithWinBackOffer", void 0);
      }
      purchasePackageWithWinBackOffer(_options) {
        return this.mockReturningFunctionIfEnabled("purchasePackageWithWinBackOffer", void 0);
      }
      invalidateCustomerInfoCache() {
        return this.mockNonReturningFunctionIfEnabled("invalidateCustomerInfoCache");
      }
      presentCodeRedemptionSheet() {
        return this.mockNonReturningFunctionIfEnabled("presentCodeRedemptionSheet");
      }
      setAttributes(_attributes) {
        return this.mockNonReturningFunctionIfEnabled("setAttributes");
      }
      setEmail(_email) {
        return this.mockNonReturningFunctionIfEnabled("setEmail");
      }
      setPhoneNumber(_phoneNumber) {
        return this.mockNonReturningFunctionIfEnabled("setPhoneNumber");
      }
      setDisplayName(_displayName) {
        return this.mockNonReturningFunctionIfEnabled("setDisplayName");
      }
      setPushToken(_pushToken) {
        return this.mockNonReturningFunctionIfEnabled("setPushToken");
      }
      setProxyURL(_url) {
        return this.mockNonReturningFunctionIfEnabled("setProxyURL");
      }
      collectDeviceIdentifiers() {
        return this.mockNonReturningFunctionIfEnabled("collectDeviceIdentifiers");
      }
      setAdjustID(_adjustID) {
        return this.mockNonReturningFunctionIfEnabled("setAdjustID");
      }
      setAppsflyerID(_appsflyerID) {
        return this.mockNonReturningFunctionIfEnabled("setAppsflyerID");
      }
      setFBAnonymousID(_fbAnonymousID) {
        return this.mockNonReturningFunctionIfEnabled("setFBAnonymousID");
      }
      setMparticleID(_mparticleID) {
        return this.mockNonReturningFunctionIfEnabled("setMparticleID");
      }
      setCleverTapID(_cleverTapID) {
        return this.mockNonReturningFunctionIfEnabled("setCleverTapID");
      }
      setMixpanelDistinctID(_mixpanelDistinctID) {
        return this.mockNonReturningFunctionIfEnabled("setMixpanelDistinctID");
      }
      setFirebaseAppInstanceID(_firebaseAppInstanceID) {
        return this.mockNonReturningFunctionIfEnabled("setFirebaseAppInstanceID");
      }
      setOnesignalID(_onesignalID) {
        return this.mockNonReturningFunctionIfEnabled("setOnesignalID");
      }
      setOnesignalUserID(_onesignalUserID) {
        return this.mockNonReturningFunctionIfEnabled("setOnesignalUserID");
      }
      setSingularDeviceID(_singularDeviceID) {
        return this.mockNonReturningFunctionIfEnabled("setSingularDeviceID");
      }
      setAirshipChannelID(_airshipChannelID) {
        return this.mockNonReturningFunctionIfEnabled("setAirshipChannelID");
      }
      setMediaSource(_mediaSource) {
        return this.mockNonReturningFunctionIfEnabled("setMediaSource");
      }
      setCampaign(_campaign) {
        return this.mockNonReturningFunctionIfEnabled("setCampaign");
      }
      setAdGroup(_adGroup) {
        return this.mockNonReturningFunctionIfEnabled("setAdGroup");
      }
      setAd(_ad) {
        return this.mockNonReturningFunctionIfEnabled("setAd");
      }
      setKeyword(_keyword) {
        return this.mockNonReturningFunctionIfEnabled("setKeyword");
      }
      setCreative(_creative) {
        return this.mockNonReturningFunctionIfEnabled("setCreative");
      }
      canMakePayments(_features) {
        return this.mockReturningFunctionIfEnabled("canMakePayments", {
          canMakePayments: true
        });
      }
      beginRefundRequestForActiveEntitlement() {
        const mockResult = {
          refundRequestStatus: REFUND_REQUEST_STATUS.USER_CANCELLED
        };
        return this.mockReturningFunctionIfEnabled("beginRefundRequestForActiveEntitlement", mockResult);
      }
      beginRefundRequestForEntitlement(_entitlementInfo) {
        const mockResult = {
          refundRequestStatus: REFUND_REQUEST_STATUS.USER_CANCELLED
        };
        return this.mockReturningFunctionIfEnabled("beginRefundRequestForEntitlement", mockResult);
      }
      beginRefundRequestForProduct(_storeProduct) {
        const mockResult = {
          refundRequestStatus: REFUND_REQUEST_STATUS.USER_CANCELLED
        };
        return this.mockReturningFunctionIfEnabled("beginRefundRequestForProduct", mockResult);
      }
      showInAppMessages(_options) {
        return this.mockNonReturningFunctionIfEnabled("showInAppMessages");
      }
      isConfigured() {
        const mockResult = { isConfigured: true };
        return this.mockReturningFunctionIfEnabled("isConfigured", mockResult);
      }
      overridePreferredUILocale(_options) {
        return this.mockNonReturningFunctionIfEnabled("overridePreferredUILocale");
      }
      getVirtualCurrencies() {
        return this.mockReturningFunctionIfEnabled("getVirtualCurrencies", {
          virtualCurrencies: this.mockEmptyVirtualCurrencies
        });
      }
      invalidateVirtualCurrenciesCache() {
        return this.mockNonReturningFunctionIfEnabled("invalidateVirtualCurrenciesCache");
      }
      getCachedVirtualCurrencies() {
        return this.mockReturningFunctionIfEnabled("getCachedVirtualCurrencies", {
          cachedVirtualCurrencies: this.mockEmptyVirtualCurrencies
        });
      }
      trackCustomPaywallImpression(_options) {
        return this.mockNonReturningFunctionIfEnabled("trackCustomPaywallImpression");
      }
      mockTransaction(productIdentifier) {
        return {
          productIdentifier,
          purchaseDate: (/* @__PURE__ */ new Date()).toISOString(),
          transactionIdentifier: "",
          purchaseToken: null,
          originalJson: null,
          signature: null
        };
      }
      mockNonReturningFunctionIfEnabled(functionName) {
        if (!this.shouldMockWebResults) {
          return Promise.reject(this.webNotSupportedErrorMessage);
        }
        console.log(`${functionName} called on web with mocking enabled. No-op`);
        return Promise.resolve();
      }
      mockReturningFunctionIfEnabled(functionName, returnValue) {
        if (!this.shouldMockWebResults) {
          return Promise.reject(this.webNotSupportedErrorMessage);
        }
        console.log(`${functionName} called on web with mocking enabled. Returning mocked value`);
        return Promise.resolve(returnValue);
      }
    };
  }
});

// node_modules/@revenuecat/purchases-capacitor/dist/esm/index.js
init_dist();
init_dist2();

// node_modules/@revenuecat/purchases-capacitor/dist/esm/definitions.js
init_dist2();

// node_modules/@revenuecat/purchases-capacitor/dist/esm/index.js
var nativePlugin = registerPlugin("Purchases", {
  web: () => Promise.resolve().then(() => (init_web(), web_exports)).then((m) => new m.PurchasesWeb())
});
function normalizeRejection(result) {
  if (!(result instanceof Promise)) {
    return result;
  }
  const normalized = result.then(void 0, (error) => {
    throw normalizePurchasesError(error);
  });
  const remove = result.remove;
  if (typeof remove === "function") {
    normalized.remove = remove;
  }
  return normalized;
}
function normalizeRedemptionError(result) {
  if (!(result instanceof Promise)) {
    return result;
  }
  return result.then((redemption) => {
    normalizePurchasesError(redemption === null || redemption === void 0 ? void 0 : redemption.error);
    return redemption;
  });
}
function getNativeTrackCustomPaywallImpressionOptions(options) {
  var _a;
  const offering = options === null || options === void 0 ? void 0 : options.offering;
  const nativeOptions = {};
  if ((options === null || options === void 0 ? void 0 : options.paywallId) != null) {
    nativeOptions.paywallId = options.paywallId;
  }
  if (offering != null) {
    nativeOptions.offeringId = offering.identifier;
    const presentedOfferingContext = (_a = offering.availablePackages[0]) === null || _a === void 0 ? void 0 : _a.presentedOfferingContext;
    if (presentedOfferingContext != null) {
      nativeOptions.presentedOfferingContext = presentedOfferingContext;
    }
  } else if ((options === null || options === void 0 ? void 0 : options.offeringId) != null) {
    nativeOptions.offeringId = options.offeringId;
  }
  return nativeOptions;
}
function toNativeArgs(prop, args) {
  if (prop !== "trackCustomPaywallImpression") {
    return args;
  }
  return [getNativeTrackCustomPaywallImpressionOptions(args[0])];
}
var Purchases = new Proxy(nativePlugin, {
  get(target, prop, receiver) {
    const value = Reflect.get(target, prop, receiver);
    if (typeof value !== "function") {
      return value;
    }
    return new Proxy(value, {
      apply: (method, _thisArg, args) => {
        const result = normalizeRejection(Reflect.apply(method, target, toNativeArgs(prop, args)));
        return prop === "redeemWebPurchase" ? normalizeRedemptionError(result) : result;
      }
    });
  }
});

// webrtc/sender/secure/subscription.js
var ENTITLEMENT = "premium";
var MOCK_PRICE = "\xA32.99";
var RENEW_TEXT = "\nRenews automatically unless cancelled at least 24 hours before the end of the current period. Manage or cancel in your App Store settings.";
var isNativeIos = () => !!window.Capacitor?.isNativePlatform?.() && window.Capacitor.getPlatform() === "ios";
var MOCK_KEY = "submock";
var qsMock = new URLSearchParams(location.search).get(MOCK_KEY);
if (!isNativeIos() && qsMock) {
  try {
    if (qsMock === "off") {
      localStorage.removeItem(MOCK_KEY);
    } else {
      localStorage.setItem(MOCK_KEY, qsMock);
    }
  } catch {
  }
}
var mockState = () => isNativeIos() ? null : localStorage.getItem(MOCK_KEY);
var setMock = (v) => {
  try {
    localStorage.setItem(MOCK_KEY, v);
  } catch {
  }
};
if (mockState()) {
  console.warn("[SUB] TEST MODE:", mockState());
}
var needsSubscription = () => isNativeIos() || !!mockState();
function diag(message) {
  const text = `[RC] ${message}`;
  console.log(text);
  try {
    window.dispatchEvent(
      new CustomEvent("naptio:subscription-debug", {
        detail: text
      })
    );
  } catch {
  }
}
var ready = null;
async function rc() {
  if (!isNativeIos()) {
    throw new Error("RevenueCat is available only in the native iOS app.");
  }
  if (!ready) {
    const startedAt = performance.now();
    diag("configure START");
    ready = (async () => {
      diag("setLogLevel START");
      await Purchases.setLogLevel({
        level: LOG_LEVEL.DEBUG
      });
      diag("setLogLevel OK");
      diag("Purchases.configure START");
      diag(
        `Purchases.configure OK after ${Math.round(
          performance.now() - startedAt
        )} ms`
      );
      diag("Purchases.configure OK");
    })().catch((err) => {
      ready = null;
      diag(`configure ERROR: ${err?.message || String(err)}`);
      console.error("[SUB] RevenueCat configuration failed", err);
      throw err;
    });
  } else {
    diag("configure: using existing request");
  }
  await ready;
  diag("RevenueCat configured");
  return Purchases;
}
var isActive = (info) => !!info?.entitlements?.active?.[ENTITLEMENT];
function withTimeout(promise, timeoutMs, label) {
  return Promise.race([
    promise,
    new Promise(
      (_, reject) => setTimeout(
        () => reject(
          new Error(`${label} timed out after ${timeoutMs} ms`)
        ),
        timeoutMs
      )
    )
  ]);
}
async function hasAccess() {
  if (mockState()) {
    return mockState() === "active";
  }
  if (!needsSubscription()) {
    return true;
  }
  try {
    diag("access check START");
    const purchases = await withTimeout(
      rc(),
      3e4,
      "RevenueCat initialization"
    );
    diag("CustomerInfo START");
    const { customerInfo } = await withTimeout(
      purchases.getCustomerInfo(),
      1e4,
      "RevenueCat customer-info request"
    );
    diag("CustomerInfo OK");
    const active = isActive(customerInfo);
    diag(`premium active: ${active}`);
    return active;
  } catch (e) {
    diag(`access ERROR: ${e?.message || String(e)}`);
    console.warn("[SUB] access check failed", e);
    return false;
  }
}
async function getPackage() {
  diag("getPackage START");
  const purchases = await rc();
  diag("Offerings START");
  const offerings = await purchases.getOfferings();
  diag("Offerings OK");
  const cur = offerings?.current;
  if (!cur) {
    diag("Offering ERROR: no current offering");
    throw new Error(
      "No current RevenueCat offering is configured."
    );
  }
  if (!cur.monthly) {
    diag("Offering ERROR: monthly package unavailable");
    throw new Error(
      "The monthly subscription package is unavailable."
    );
  }
  diag("Monthly package OK");
  return cur.monthly;
}
function buildOffer(priceString, trialText) {
  return trialText ? {
    title: "Start your free trial",
    terms: `${trialText} free, then ${priceString} per month. ${RENEW_TEXT}`,
    button: "Start free trial"
  } : {
    title: "Subscribe",
    terms: `${priceString} per month. ${RENEW_TEXT}`,
    button: `Subscribe for ${priceString}/month`
  };
}
async function offer() {
  if (mockState()) {
    return buildOffer(
      MOCK_PRICE,
      mockState() === "new" ? "2 weeks" : null
    );
  }
  diag("offer START");
  const pkg = await getPackage();
  const p = pkg.product;
  diag(`Product OK: ${p?.identifier || "unknown"}`);
  diag(`Price OK: ${p?.priceString || "unknown"}`);
  const intro = p.introPrice;
  let trialText = null;
  if (intro && intro.price === 0) {
    diag("Introductory offer detected");
    diag("Eligibility START");
    let eligible = false;
    try {
      const purchases = await rc();
      const res = await purchases.checkTrialOrIntroductoryPriceEligibility({
        productIdentifiers: [p.identifier]
      });
      eligible = res?.[p.identifier]?.status === 2;
      diag(`Eligibility OK: ${eligible}`);
    } catch (e) {
      diag(
        `Eligibility ERROR: ${e?.message || String(e)}`
      );
      console.warn(
        "[SUB] eligibility check failed",
        e
      );
    }
    if (eligible) {
      const n = intro.periodNumberOfUnits;
      trialText = `${n} ${intro.periodUnit.toLowerCase()}${n > 1 ? "s" : ""}`;
      diag(`Trial available: ${trialText}`);
    }
  } else {
    diag("No free introductory offer");
  }
  diag("offer OK");
  return buildOffer(
    p.priceString,
    trialText
  );
}
async function purchase() {
  if (mockState()) {
    if (!confirm(
      "[TEST] Simulated Apple purchase sheet.\n\nOK = subscribe, Cancel = close"
    )) {
      return false;
    }
    setMock("active");
    return true;
  }
  diag("purchase START");
  const pkg = await getPackage();
  try {
    diag("purchasePackage START");
    const purchases = await rc();
    const { customerInfo } = await purchases.purchasePackage({
      aPackage: pkg
    });
    diag("purchasePackage OK");
    const active = isActive(customerInfo);
    diag(`premium active after purchase: ${active}`);
    return active;
  } catch (e) {
    if (e?.userCancelled || e?.code === "1") {
      diag("purchase CANCELLED");
      return false;
    }
    diag(
      `purchase ERROR: ${e?.message || String(e)}`
    );
    throw e;
  }
}
async function restore() {
  if (mockState()) {
    return mockState() === "active";
  }
  diag("restore START");
  try {
    const purchases = await rc();
    diag("restorePurchases START");
    const { customerInfo } = await purchases.restorePurchases();
    diag("restorePurchases OK");
    const active = isActive(customerInfo);
    diag(`premium active after restore: ${active}`);
    return active;
  } catch (e) {
    diag(
      `restore ERROR: ${e?.message || String(e)}`
    );
    throw e;
  }
}
export {
  hasAccess,
  needsSubscription,
  offer,
  purchase,
  restore
};
/*! Bundled license information:

@capacitor/core/dist/index.js:
  (*! Capacitor: https://capacitorjs.com/ - MIT License *)
*/
