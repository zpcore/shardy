/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This test demonstrates how Shardy's sharding propagation works for reshape
// operations, specifically showing how axes are split into sub-axes when
// projecting dimension shardings to factor shardings.
//
// Example scenario:
//   Reshape: A [ij, k] -> B [i, jk] with factors i=8, j=2, k=4
//   Tensor A shape: [16, 4] (since ij=8*2=16)
//   Tensor B shape: [8, 8] (since i=8, jk=2*4=8)
//
// When A is sharded as [{"x", "y"}, {}] with mesh <"x"=8, "y"=4>:
//   - Dimension 0 of A (size 16) is sharded by "x" (size 8) then "y" (size 4)
//   - Total sharding = 8 * 4 = 32, but dim size is 16, so only part is used
//
// The sharding projection splits axes across factors:
//   - Factor i (size 8): gets "x" (full axis, size 8) -> total 8 ✓
//   - Factor j (size 2): gets "y":(1)2 (first half of "y", size 2) -> total 2 ✓
//   - Factor k (size 4): unsharded (dimension 1 of A is unsharded)

#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/testing_utils.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

inline constexpr StringRef kMeshName = "mesh";

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

MeshAttr getMeshAttr(ModuleOp module) {
  return cast<MeshOp>(module.lookupSymbol(kMeshName)).getMesh();
}

template <class OpTy>
OpTy getFirstOp(ModuleOp module) {
  auto mainFn = cast<func::FuncOp>(module.lookupSymbol("main"));
  auto ops = mainFn.getBody().front().getOps<OpTy>();
  assert(!ops.empty());
  return *ops.begin();
}

class ReshapePropagationDemoTest : public ShardyTestBase {};

// =============================================================================
// Test 1: Basic reshape split dimension - demonstrates sub-axis creation
//
// Reshape: [8] -> [2, 4]
// Sharding rule factors: factor_0 = 2, factor_1 = 4
// Mesh: <"a"=4, "b"=2>
//
// Input tensor is sharded by ["a", "b"] on dimension 0.
// Total sharding size = 4 * 2 = 8, which matches dimension size.
//
// Factor projection:
//   - factor_0 (size 2): needs 2 positions from the sharding
//     -> gets "a":(1)2 (first sub-axis of "a", pre_size=1, size=2)
//   - factor_1 (size 4): needs 4 positions
//     -> gets "a":(2)2 (second sub-axis of "a", pre_size=2, size=2) + "b" (full)
//     -> total = 2 * 2 = 4 ✓
// =============================================================================
TEST_F(ReshapePropagationDemoTest, ReshapeSplitDimWithSubAxes) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  MeshAttr mesh = getMeshAttr(module.get());

  // Print the sharding rule for understanding
  llvm::outs() << "\n=== Test 1: ReshapeSplitDimWithSubAxes ===\n";
  llvm::outs() << "Sharding Rule: " << shardingRule << "\n";
  llvm::outs() << "Factor sizes: [";
  for (int64_t size : shardingRule.getFactorSizes()) {
    llvm::outs() << size << ", ";
  }
  llvm::outs() << "]\n";

  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);

  // Print the factor shardings
  llvm::outs() << "\nOperand 0 factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getOperand(0).factorIndexToSharding) {
    llvm::outs() << "  Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] isClosed=" << factorSharding.isClosed
                 << " isMinorMost=" << factorSharding.isMinorMost << "\n";
  }

  // Verify the projection
  // Factor 0 (size 2): gets sub-axis "a":(1)2
  // Factor 1 (size 4): gets sub-axis "a":(2)2 and full axis "b"
  EXPECT_THAT(
      projection.getOperand(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                               /*isMinorMost*/ false,
                               ElementsAre(SubAxisRefIs("a", 1, 2))),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(SubAxisRefIs("a", 2, 2),
                                           AxisRefIs("b")))),
          /*replicatedAxes*/ IsEmpty(),
          /*unreducedAxes*/ IsEmpty()));

  // Result has no sharding (open factors)
  EXPECT_THAT(
      projection.getResult(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                               /*isMinorMost*/ true, IsEmpty()),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ false,
                               /*isMinorMost*/ true, IsEmpty())),
          /*replicatedAxes*/ IsEmpty(),
          /*unreducedAxes*/ IsEmpty()));
}

// =============================================================================
// Test 2: Your example - A [ij, k] -> B [i, jk]
//
// Reshape: [16, 4] -> [8, 8] with factors i=8, j=2, k=4
// Mesh: <"x"=8, "y"=4>
//
// Input A is sharded as [{"x", "y"}, {}]
//   - Dimension 0 (size 16 = i*j) sharded by "x"*"y" = 8*4 = 32
//   - But we can only use up to size 16, so effectively "x"(8) + "y":(1)2
//
// Factor projection for operand A (shape [ij, k]):
//   - Factor i (size 8): gets "x" (full axis)
//   - Factor j (size 2): gets "y":(1)2 (first half of "y")
//   - Factor k (size 4): unsharded (dimension 1 has no sharding)
// =============================================================================
TEST_F(ReshapePropagationDemoTest, ReshapeMergeAndSplitFactors) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["x"=8, "y"=4]>

    func.func @main(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>})
        -> tensor<8x8xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<16x4xf32>) -> tensor<8x8xf32>
      return %0 : tensor<8x8xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  MeshAttr mesh = getMeshAttr(module.get());

  llvm::outs() << "\n=== Test 2: ReshapeMergeAndSplitFactors ===\n";
  llvm::outs() << "Input shape: [16, 4], Output shape: [8, 8]\n";
  llvm::outs() << "Sharding Rule: " << shardingRule << "\n";
  llvm::outs() << "Factor sizes: [";
  for (int64_t size : shardingRule.getFactorSizes()) {
    llvm::outs() << size << ", ";
  }
  llvm::outs() << "]\n";

  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);

  llvm::outs() << "\nOperand factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getOperand(0).factorIndexToSharding) {
    llvm::outs() << "  Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] isClosed=" << factorSharding.isClosed
                 << " isMinorMost=" << factorSharding.isMinorMost << "\n";
  }

  llvm::outs() << "\nResult factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getResult(0).factorIndexToSharding) {
    llvm::outs() << "  Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] isClosed=" << factorSharding.isClosed
                 << " isMinorMost=" << factorSharding.isMinorMost << "\n";
  }
}

// =============================================================================
// Test 2.1: Your example - A [ij, k] -> B [i, jk]
//
// Reshape: [16, 4] -> [8, 8] with factors i=8, j=2, k=4
// Mesh: <"x"=4, "y"=4>
//
// Input A is sharded as [{"x", "y"}, {}]
//   - Dimension 0 (size 16 = i*j) sharded by "x"*"y" = 8*4 = 32
//   - But we can only use up to size 16, so effectively "x"(4) + "y":(1)2
//
// Factor projection for operand A (shape [ij, k]):
//   - Factor i (size 8): gets "x" (full axis)
//   - Factor j (size 2): gets "y":(1)2 (first half of "y")
//   - Factor k (size 4): unsharded (dimension 1 has no sharding)
// =============================================================================
TEST_F(ReshapePropagationDemoTest, ReshapeMergeAndSplitFactors2) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["x"=4, "y"=4]>

    func.func @main(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>})
        -> tensor<8x8xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<16x4xf32>) -> tensor<8x8xf32>
      return %0 : tensor<8x8xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  MeshAttr mesh = getMeshAttr(module.get());

  llvm::outs() << "\n=== Test 2.5: ReshapeMergeAndSplitFactors ===\n";
  llvm::outs() << "Input shape: [16, 4], Output shape: [8, 8]\n";
  llvm::outs() << "Sharding Rule: " << shardingRule << "\n";
  llvm::outs() << "Factor sizes: [";
  for (int64_t size : shardingRule.getFactorSizes()) {
    llvm::outs() << size << ", ";
  }
  llvm::outs() << "]\n";

  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);

  llvm::outs() << "\nOperand factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getOperand(0).factorIndexToSharding) {
    llvm::outs() << "  Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] isClosed=" << factorSharding.isClosed
                 << " isMinorMost=" << factorSharding.isMinorMost << "\n";
  }

  llvm::outs() << "\nResult factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getResult(0).factorIndexToSharding) {
    llvm::outs() << "  Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] isClosed=" << factorSharding.isClosed
                 << " isMinorMost=" << factorSharding.isMinorMost << "\n";
  }
}



// =============================================================================
// Test 3: Reshape with open sharding (?) - shows propagation potential
//
// When a dimension has open sharding [{"x", ?}], the factor sharding
// is also open (isClosed=false), indicating more axes can be added.
// =============================================================================
TEST_F(ReshapePropagationDemoTest, ReshapeWithOpenSharding) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["x"=4, "y"=2]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", ?}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  MeshAttr mesh = getMeshAttr(module.get());

  llvm::outs() << "\n=== Test 3: ReshapeWithOpenSharding ===\n";
  llvm::outs() << "Input sharding: [{\"x\", ?}] - open, can add more axes\n";

  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);

  llvm::outs() << "\nOperand factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getOperand(0).factorIndexToSharding) {
    llvm::outs() << "  Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] isClosed=" << factorSharding.isClosed
                 << " isMinorMost=" << factorSharding.isMinorMost << "\n";
  }

  // Factor 0 (size 2): gets "x":(1)2 (first half of "x"), and is OPEN
  // Factor 1 (size 4): gets "x":(2)2 (second half of "x"), and is OPEN
  EXPECT_THAT(
      projection.getOperand(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                               /*isMinorMost*/ false,
                               ElementsAre(SubAxisRefIs("x", 1, 2))),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ false,
                               /*isMinorMost*/ true,
                               ElementsAre(SubAxisRefIs("x", 2, 2)))),
          /*replicatedAxes*/ IsEmpty(),
          /*unreducedAxes*/ IsEmpty()));
}

// =============================================================================
// Test 4: Demonstrates propagation - expanding sharding from operand to result
//
// This test shows how the ShardingProjection::expandSharding method can
// propagate factor shardings from an operand to a result.
// =============================================================================
TEST_F(ReshapePropagationDemoTest, PropagateFactorShardings) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["x"=2, "y"=2]>

    func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>})
        -> tensor<16xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
      return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  MeshAttr mesh = getMeshAttr(module.get());

  llvm::outs() << "\n=== Test 4: PropagateFactorShardings ===\n";
  llvm::outs() << "Demonstrates how sharding propagates through reshape\n";
  llvm::outs() << "Sharding Rule: " << shardingRule << "\n";

  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);

  llvm::outs() << "\nBefore propagation:\n";
  llvm::outs() << "  Operand factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getOperand(0).factorIndexToSharding) {
    llvm::outs() << "    Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "]\n";
  }
  llvm::outs() << "  Result factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getResult(0).factorIndexToSharding) {
    llvm::outs() << "    Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "] (empty means open/unsharded)\n";
  }

  // Now propagate: get the greatest common prefix axes and expand
  AxesPerFactor gcpAxes =
      projection.getGreatestCommonPrefixAxes(shardingRule.getFactorSizes().size());

  llvm::outs() << "\nGreatest Common Prefix axes per factor:\n";
  for (size_t i = 0; i < gcpAxes.size(); ++i) {
    llvm::outs() << "  Factor " << i << ": [";
    for (AxisRefAttr axis : gcpAxes[i]) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "]\n";
  }

  // Expand result shardings to match operand
  for (int64_t factorIdx = 0;
       factorIdx < static_cast<int64_t>(gcpAxes.size()); ++factorIdx) {
    if (!gcpAxes[factorIdx].empty()) {
      projection.expandSharding(factorIdx, gcpAxes[factorIdx]);
    }
  }

  llvm::outs() << "\nAfter propagation (expandSharding):\n";
  llvm::outs() << "  Result factor shardings:\n";
  for (const auto& [factorIdx, factorSharding] :
       projection.getResult(0).factorIndexToSharding) {
    llvm::outs() << "    Factor " << factorIdx << ": [";
    for (AxisRefAttr axis : factorSharding.axisRefs) {
      llvm::outs() << axis.toString() << ", ";
    }
    llvm::outs() << "]\n";
  }

  // Reconstruct the result tensor sharding from the projection
  TensorShardingAttr reconstructedResultSharding =
      projection.getResult(0).createTensorShardingAttr(
          &context, shardingRule.getResultMappings()[0],
          shardingRule.getFactorSizes(), kMeshName, mesh);

  llvm::outs() << "\nReconstructed result sharding: "
               << reconstructedResultSharding << "\n";
}

// =============================================================================
// Test 5: Sub-axis notation explained with concrete example
//
// This test provides a detailed walkthrough of what sub-axis notation means.
// =============================================================================
TEST_F(ReshapePropagationDemoTest, SubAxisNotationExplained) {
  llvm::outs() << "\n=== Test 5: Sub-Axis Notation Explained ===\n\n";

  // Create a mesh with axis "y" of size 4
  MeshAttr mesh = createMesh({{"y", 4}});
  llvm::outs() << "Mesh: <\"y\"=4>\n";
  llvm::outs() << "Axis \"y\" represents 4 devices: [0, 1, 2, 3]\n\n";

  // Sub-axis "y":(1)2 means:
  //   pre_size = 1 (start from beginning)
  //   size = 2 (take 2 elements)
  // This corresponds to devices [0, 1]
  AxisRefAttr subAxis1 = createSubAxis("y", 1, 2);
  llvm::outs() << "Sub-axis \"y\":(1)2:\n";
  llvm::outs() << "  pre_size = 1 (product of sizes to the left = 1, i.e., major-most)\n";
  llvm::outs() << "  size = 2\n";
  llvm::outs() << "  Represents: first 2 elements [0, 1]\n";
  llvm::outs() << "  Axis size in sharding: " << subAxis1.getSize(mesh) << "\n\n";

  // Sub-axis "y":(2)2 means:
  //   pre_size = 2 (skip first 2 positions worth of elements)
  //   size = 2 (take 2 elements)
  // This corresponds to devices [2, 3]
  AxisRefAttr subAxis2 = createSubAxis("y", 2, 2);
  llvm::outs() << "Sub-axis \"y\":(2)2:\n";
  llvm::outs() << "  pre_size = 2 (product of sizes to the left = 2, skip first sub-axis)\n";
  llvm::outs() << "  size = 2\n";
  llvm::outs() << "  Represents: next 2 elements [2, 3]\n";
  llvm::outs() << "  Axis size in sharding: " << subAxis2.getSize(mesh) << "\n\n";

  // Full axis vs sub-axes
  AxisRefAttr fullAxis = createAxis("y");
  llvm::outs() << "Full axis \"y\":\n";
  llvm::outs() << "  Represents all 4 elements [0, 1, 2, 3]\n";
  llvm::outs() << "  Axis size in sharding: " << fullAxis.getSize(mesh) << "\n\n";

  llvm::outs() << "Key insight: When a dimension is split by reshape, the axes\n";
  llvm::outs() << "sharding that dimension are also split into sub-axes that\n";
  llvm::outs() << "correspond to each factor of the reshape.\n";
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
