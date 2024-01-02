// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "extension_mngr.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 ExtensionManager::Ptr extensionManager,
                 WeightsSharing::Ptr w_cache,
                 bool isGraphQuantized,
                 ov::threading::IStreamsExecutor::Ptr streamExecutor = nullptr)
        : config(config),
          extensionManager(extensionManager),
          weightsCache(w_cache),
          streamExecutor(streamExecutor),
          isGraphQuantizedFlag(isGraphQuantized) {
        rtParamsCache = std::make_shared<MultiCache>(config.rtCacheCapacity);

        // primitive/executors can be shared across sub-stream
        // but scratch pad cannot be shared.
        numSubStreams = 1;
        if (streamExecutor) {
            auto cpuStreamExecutor = std::dynamic_pointer_cast<ov::threading::CPUStreamsExecutor>(streamExecutor);
            auto nNumaNodes = static_cast<int>(cpuStreamExecutor->get_cores_mt_sockets().size());
            if (numSubStreams < nNumaNodes)
                numSubStreams = nNumaNodes;
        }
        for (int i = 0; i < numSubStreams; i++) {
            rtScratchPads.push_back(std::make_shared<DnnlScratchPad>(getEngine(), i));
        }
    }

    const Config& getConfig() const {
        return config;
    }

    ExtensionManager::Ptr getExtensionManager() const {
        return extensionManager;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return weightsCache;
    }


    MultiCachePtr getParamsCache() const {
        return rtParamsCache;
    }

    DnnlScratchPadPtr getScratchPad(int subStreamID = 0) const {
        return rtScratchPads[subStreamID];
    }

    static const dnnl::engine& getEngine();

    bool isGraphQuantized() const {
        return isGraphQuantizedFlag;
    }

    ov::threading::IStreamsExecutor::Ptr getStreamExecutor() const {
        return streamExecutor;
    }

    int getNumSubStreams() const {
        return numSubStreams;
    }

private:
    Config config;  // network-level config

    ExtensionManager::Ptr extensionManager;
    WeightsSharing::Ptr weightsCache;         // per NUMA node caches for sharing weights data

    MultiCachePtr rtParamsCache;     // primitive cache
    std::vector<DnnlScratchPadPtr> rtScratchPads;  // scratch pad (each sub-stream has its own copy)

    ov::threading::IStreamsExecutor::Ptr streamExecutor;   // stream executor for current graph

    int numSubStreams;

    bool isGraphQuantizedFlag = false;
};

}  // namespace intel_cpu
}  // namespace ov
