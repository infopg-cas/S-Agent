import { useCallback } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  useNodesState,
  useEdgesState,
  MiniMap,
  Controls,
} from 'reactflow';
import 'reactflow/dist/style.css';

const initialNodes = [
  {
    id: '1',
    type: 'input',
    sourcePosition: 'right',
    data: { label: '人类用户' },
    position: { x: 1, y: 200 },
    className: 'light',
  },
  {
    id: '2',
    data: { label: '管理会（Meta Group)' },
    position: { x: 200, y: 100 },
    className: 'light',
    style: { backgroundColor: 'rgba(255, 0, 0, 0.2)', width: 200, height: 250 },
  },
  {
    id: '2a',
    data: { label: '任务规划智能体' },
    sourcePosition: 'right',
    position: { x: 20, y: 50 },
    parentId: '2',
  },
  {
    id: '2b',
    data: { label: '任务决策智能体' },
    position: { x: 20, y: 100 },
    parentId: '2',
  },

  {
    id: '2c',
    data: { label: '内容审核智能体' },
    targetPosition: 'right',
    position: { x: 20, y: 150 },
    parentId: '2',
  },
  {
    id: '2d',
    data: { label: '人机交互智能体' },
    targetPosition: 'bottom',
    position: { x: 20, y: 200 },
    parentId: '2',
  },

  {
    id: '4',
    data: { label: '工作组Woker Group' },
    position: { x: 420, y: 80 },
    targetPosition: 'left',
    className: 'light',
    style: { backgroundColor: 'rgba(255, 0, 0, 0.2)', width: 300, height: 300 },
  },
  {
    id: '4a',
    data: { label: 'RAG智能体' },
    position: { x: 65, y: 45 },
    className: 'light',
    parentId: '4',
    extent: 'parent',
  },
  {
    id: '4b',
    data: { label: '数据自动化组' },
    position: { x: 15, y: 120 },
    className: 'light',
    style: {
      backgroundColor: 'rgba(255, 0, 255, 0.2)',
      height: 150,
      width: 270,
    },
    parentId: '4',
  },
  {
    id: '4b1',
    data: { label: '数据库智能体' },
    position: { x: 20, y: 40 },
    className: 'light',
    parentId: '4b',
  },
  {
    id: '4b2',
    data: { label: '数据分析智能体' },
    position: { x: 100, y: 100 },
    className: 'light',
    parentId: '4b',
  },
];

const initialEdges = [
  { id: 'e1-2', source: '1', target: '2d', animated: true,  markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: 'e1-3', source: '1', target: '3', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: 'e2a-4', source: '2a', target: '4', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: 'e3-4b', source: '3', target: '4b', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: 'e4a-4b1', source: '4a', target: '4b1', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: 'e4a-4b2', source: '4a', target: '4b2', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: 'e4b1-4b2', source: '4b1', target: '4b2', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: '4b2-2d', source: '4b2', target: '2d', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: '4a-2c', source: '4a', target: '2c', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: '4b1-2c', source: '4b1', target: '2c', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
  { id: '4b1-2c', source: '4b2', target: '2c', markerStart: 'myCustomSvgMarker', markerEnd: { type: 'arrow', color: '#f00' }, },
];

const NestedFlow = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((connection) => {
    setEdges((eds) => addEdge(connection, eds));
  }, []);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      className="react-flow-subflows-example"
      fitView
      attributionPosition="bottom-left"
    >
      <MiniMap />
      <Controls />
      <Background />
    </ReactFlow>
  );
};

export default NestedFlow;
