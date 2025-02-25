const { getDefaultConfig } = require("metro-config");

module.exports = (async () => {
  const {
    resolver: { assetExts },
  } = await getDefaultConfig();
  return {
    transformer: {
      getTransformOptions: async () => ({
        transformOptions: {
          inlineRequires: true,
        },
      }),
    },
    resolver: {
      assetExts: [...assetExts, "tflite"], // Add .tflite to recognized extensions
    },
  };
})();
