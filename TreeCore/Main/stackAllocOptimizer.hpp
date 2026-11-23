#pragma once
//optimiced for std::_Tree_node<class Node, void* __ptr64>

#include "Settings.h"

//#define DebugPrintSAO

#include <type_traits>
//#include <cstdlib>
#include <stdlib.h>

#ifdef DebugPrintSAO
#include <iostream>
#endif

#ifdef DEBUG
	using SetElements = std::_Container_proxy;
	const size_t SetElementSizeMax = sizeof(SetElements);
	const size_t DeallocStartOffset = SetElementSizeMax;
#else
	using SetElements = std::_Tree_node<class Node, void* __ptr64>;
	const size_t SetElementSizeMax = sizeof(SetElements);
	const size_t DeallocStartOffset = SetElementSizeMax;
#endif

//x64 intel,amd and arm use 64 B line size for cache so for best performace allign to that or optimze to case by case but std::_Tree_node<class Node, void* __ptr64> is 128 Byte so this is the new allining

template <class T>
class stack_allocator
{
public:


	using _From_primary = stack_allocator;
	using value_type = T;

	using size_type = size_t;
	using difference_type = ptrdiff_t;
	using pointer = T*;
	using const_pointer = const T*;
	using void_pointer = void*;
	using const_void_pointer = const void*;

	using propagate_on_container_move_assignment = std::bool_constant<true>;
	using is_always_equal = std::false_type;


public:

	explicit stack_allocator()
		: m_begin(nullptr), m_size(0), m_stack_pointer(nullptr)
	{
#ifdef DebugPrintSAO
		std::cout << "stack_allocator " << std::hex << typeid(T).name() << "\n";
#endif
	}

	template <class U>
	stack_allocator(const stack_allocator<U>& other)
		:  m_begin(other.m_begin), m_size(other.m_size),
		m_stack_pointer(other.m_stack_pointer)
	{
#ifdef DebugPrintSAO
		std::cout << "stack_allocator(const stack_allocator<U>& other) " << typeid(T).name() << "\n";
#endif
	}
	~stack_allocator()
	{
#ifdef DebugPrintSAO
		std::cout << "~stack_allocator " << typeid(T).name() << "\n";
#endif
		destroyAll();
	}

	void destroyAll()
	{
#ifdef MSVCRT_DEL_OPTIMIZE
		auto tmp = m_stack_pointer -= SetElementSizeMax;
		// the first element is a dummy element
		auto endPtr = m_begin + SetElementSizeMax;
		while (tmp >= m_begin + DeallocStartOffset && m_size != 0)
		{
#ifdef DebugPrintSAO
			std::cout << "~T " << static_cast<void*>(tmp) << "\n";
#endif
			auto TNode = reinterpret_cast<SetElements*>(tmp);
			TNode->_Myval.~Node();
			tmp -= SetElementSizeMax;
		}
#endif
	}

	void reserve(unsigned int size)
	{
#ifdef DebugPrintSAO
		std::cout << "reserve " << typeid(T).name() << "\n";
#endif
		destroyAll();
		if (size != m_size)
		{
			free(m_begin);
			m_begin = nullptr;
			m_begin = (char*)malloc(SetElementSizeMax * (size + 1));
			if (m_begin == nullptr)
				throw std::bad_alloc();
		}
		m_stack_pointer = reinterpret_cast<char*>(((reinterpret_cast<size_t>(m_begin) + SetElementSizeMax - 1) / SetElementSizeMax)* SetElementSizeMax);

		m_size = size;
	}

	constexpr unsigned int size() const noexcept
	{
		return static_cast<unsigned int>((m_stack_pointer - m_begin) / SetElementSizeMax);
	}

	constexpr const SetElements* ItemAt(unsigned int i) const noexcept
	{
		auto starPtr = reinterpret_cast<char*>(((reinterpret_cast<size_t>(m_begin) + SetElementSizeMax - 1) / SetElementSizeMax) * SetElementSizeMax);
		return reinterpret_cast<SetElements*>(starPtr + SetElementSizeMax * i);
	}

	constexpr T* allocate(const std::size_t n) noexcept
	{
		auto res = reinterpret_cast<T*>(m_stack_pointer);

#ifdef DebugPrintSAO
		std::cout << "alloc:" << n << " " << reinterpret_cast<size_t>(m_stack_pointer) << " " << typeid(T).name() << "\n";
#endif

		m_stack_pointer += SetElementSizeMax;
#ifdef USE_CUSTOM_ALLOC_CHECK
		if ((m_stack_pointer - m_begin) / SetElementSizeMax > m_size)
		{
			throw std::runtime_error("alloc to much:" + std::to_string(m_size));
		}
#endif
		return res;
	}
	/*
	template<typename = std::enable_if_t<
		std::is_same<T, std::_Tree_node<class Node, void* __ptr64>>::value
		>>
	constexpr T* allocate(const std::size_t n, const_void_pointer hint) noexcept
	{
#ifdef DebugPrintSAO
		std::cout << "allocHint:" << (void*)hint << " " << n << " " << typeid(T).name() << "\n";
#endif
		return allocate(n);
	}*/

	template <class U>
	struct rebind { typedef stack_allocator<U> other; };

	//do nothing
	constexpr static void deallocate(T* const p, std::size_t n) noexcept
	{
#ifdef DebugPrintSAO
		std::cout << "Dealloc:" << n << " " << (void*)p << " " << typeid(T).name() << "\n";
#endif
	}

	//just helper
	template <class _Objty, class... _Types>
	void static construct(_Objty* const _Ptr, _Types&&... _Args) {
#ifdef DebugPrintSAO
		std::cout << "new " << static_cast<void*>(_Ptr) << " " << typeid(T).name() << " " << typeid(_Objty).name() << "\n";
#endif
		::new (const_cast<void*>(static_cast<const volatile void*>(_Ptr))) _Objty(std::forward<_Types>(_Args)...);
	}

	template <class _Uty>
	void static destroy(_Uty* const _Ptr) {
#ifdef DebugPrintSAO
		std::cout << "destroy: " << (void*)_Ptr << typeid(T).name() << " " << typeid(_Uty).name() << "\n";
#endif
#ifndef MSVCRT_DEL_OPTIMIZE
		_Ptr->~_Uty();
#endif
	}

	constexpr size_type max_size() noexcept
	{
		return m_size;
	}
	void* buffer() const noexcept
	{
		return m_begin;
	}

public:

	char* m_stack_pointer;
	char* m_begin;
	unsigned int m_size;

};

template <class T1, class T2>
bool operator == (const stack_allocator<T1>& lhs,
	const stack_allocator<T2>& rhs) noexcept
{
	return lhs.buffer() == rhs.buffer();
}

template <class T1, class T2>
bool operator != (const stack_allocator<T1>& lhs,
	const stack_allocator<T2>& rhs) noexcept
{
	return !(lhs == rhs);
}