const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  ///////////////////////////////////////////////////域名访问///////////////////////////

  app.use(
    "/c1/data_api/api",
    createProxyMiddleware({
      // target:"https://10-3-21-7-60004.vpn.acas.ac.cn",
      target: "http://localhost:10986",
      changeOrigin: true,
      pathRewrite: {
        "^/c1/data_api/api": "/c1/data_api/api",
      },
    })
  );


};